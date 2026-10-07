"""오프라인 법령 검색 HTML 한 파일을 만든다.

모델(ONNX), 토크나이저, 조문 벡터, 조문 원문, onnxruntime-web(wasm)을 모두 base64로 HTML 안에 넣는다.
받은 사람은 파일을 더블클릭만 하면 된다. 인터넷·서버·API 키가 필요 없다.

    pnpm --dir apps/offline-search install          # onnxruntime-web (한 번만)
    uv run python apps/offline-search/build.py      # 모델·인덱스 준비는 README.md

조문 벡터는 반드시 HTML에 넣는 그 ONNX 모델로 만든 인덱스여야 한다
(질문은 브라우저에서 이 모델로 임베딩하므로, 다른 모델로 만든 벡터와는 비교가 안 된다).
"""

import base64
import gzip
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import cyclopts
import numpy as np
import sqlite_vec
from loguru import logger

APP_DIR = Path(__file__).parent
ROOT = APP_DIR.parents[1]
SRC = APP_DIR / "src"
ORT_DIST = APP_DIR / "node_modules" / "onnxruntime-web" / "dist"

app = cyclopts.App(help="오프라인 법령 검색 HTML 한 파일을 만든다.")


def quantize_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """행마다 배율을 따로 두는 int8 양자화. 원래 벡터 ≈ codes × scales[:, None].

    Args:
        vectors: (n, dim) float32 벡터.

    Returns:
        (codes (n, dim) int8, scales (n,) float32).
    """
    scales = np.abs(vectors).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    codes = np.round(vectors / scales[:, None]).clip(-127, 127).astype(np.int8)
    return codes, scales.astype(np.float32)


def read_index(index_path: Path) -> tuple[list[dict], np.ndarray, dict]:
    """인덱스 파일에서 조문(rowid 순)과 벡터, meta를 읽는다."""
    conn = sqlite3.connect(index_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row
    meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
    docs = [dict(row) for row in conn.execute("SELECT * FROM docs ORDER BY rowid")]
    dim = int(meta["dim"])
    blobs = conn.execute("SELECT rowid, embedding FROM vec_docs ORDER BY rowid").fetchall()
    conn.close()
    if [row["rowid"] for row in blobs] != [doc["rowid"] for doc in docs]:
        raise ValueError(f"{index_path}: docs와 vec_docs의 행이 맞지 않습니다")
    vectors = np.stack([np.frombuffer(row["embedding"], dtype=np.float32, count=dim) for row in blobs])
    return docs, vectors, meta


def corpus_payload(docs: list[dict]) -> dict:
    """화면에 필요한 열만 남긴다. 법령 정보는 법령마다 한 번만 적는다."""
    laws: dict[str, int] = {}
    law_rows = []
    rows = []
    for doc in docs:
        name = doc["law_name"]
        if name not in laws:
            laws[name] = len(law_rows)
            law_rows.append({
                "name": name,
                "type": doc["law_type"],
                "theme": doc["theme"],
                "url": doc["source_url"],
                "effectiveDate": doc["effective_date"],
            })
        rows.append([laws[name], doc["id"], doc["parent_id"] or doc["id"], doc["title"], doc["text"]])
    return {"laws": law_rows, "docs": rows}


def ort_module(dist: Path) -> tuple[str, bytes, str]:
    """onnxruntime-web을 한 <script> 안에 넣을 수 있게 고친다.

    ort.wasm.bundle.min.mjs는 wasm 로더까지 들어 있는 ES 모듈이다. 끝의 export 문을 지우고
    함수로 감싸 `const ort = {...}`로 받는다 (다른 코드와 최상위 이름이 겹치지 않게).

    Returns:
        (자바스크립트 코드, wasm 바이트, 버전).
    """
    bundle_path = dist / "ort.wasm.bundle.min.mjs"
    wasm_path = dist / "ort-wasm-simd-threaded.wasm"
    if not bundle_path.exists() or not wasm_path.exists():
        raise FileNotFoundError(
            f"onnxruntime-web이 없습니다: {dist}\n먼저 설치하세요: pnpm --dir apps/offline-search install"
        )
    code = bundle_path.read_text(encoding="utf-8")
    code = re.sub(r"\n//# sourceMappingURL=\S+\s*$", "", code)
    # 번들은 import.meta.url로 wasm 위치를 계산한다. file://에서 Firefox는 location.origin이 "null"이라
    # 이 계산이 모듈을 불러오는 순간 예외를 낸다. wasm은 바이트로 직접 넘기므로(env.wasm.wasmBinary)
    # 위치는 쓰이지 않는다. 아무 데도 없는 주소로 바꿔 둔다.
    code = code.replace("import.meta.url", '"https://offline.invalid/ort.mjs"')
    match = re.search(r"export\{([^}]*)\};?\s*$", code)
    if not match:
        raise ValueError(f"{bundle_path.name}: 끝의 export 문을 찾지 못했습니다 (onnxruntime-web 버전 확인)")
    exports = dict(reversed(item.split(" as ")) for item in match.group(1).split(","))
    missing = {"InferenceSession", "Tensor", "env"} - exports.keys()
    if missing:
        raise ValueError(f"{bundle_path.name}: export에 {missing}이 없습니다")
    names = ", ".join(f"{key}: {exports[key]}" for key in ("InferenceSession", "Tensor", "env"))
    version = json.loads((dist.parent / "package.json").read_text())["version"]
    js = f"const ort = (() => {{\n{code[: match.start()]}\nreturn {{ {names} }};\n}})();"
    return js, wasm_path.read_bytes(), version


def inline_module(path: Path) -> str:
    """src의 ES 모듈 파일을 한 <script>에 이어 붙일 수 있게 import 줄과 export 키워드를 뺀다."""
    code = path.read_text(encoding="utf-8")
    code = re.sub(r"^import .*?;\n", "", code, flags=re.MULTILINE)
    return re.sub(r"^export ", "", code, flags=re.MULTILINE)


def payload_tag(name: str, data: bytes, *, compress: bool = True) -> str:
    body = gzip.compress(data, compresslevel=9, mtime=0) if compress else data
    encoded = base64.b64encode(body).decode("ascii")
    logger.info("{:<10} 원본 {:>6.1f}MB → HTML 안 {:>6.1f}MB", name, len(data) / 1e6, len(encoded) / 1e6)
    gzip_attr = ' data-gzip="1"' if compress else ""
    return f'<script type="application/octet-stream" id="payload-{name}"{gzip_attr}>{encoded}</script>'


def model_payloads(key: str, label: str, model: Path, index: Path) -> tuple[dict, list[dict], list[str]]:
    """모델 하나의 meta 항목, 인덱스의 조문, 넣을 <script> 태그들 (토크나이저 · ONNX · 조문 벡터)."""
    onnx_path = model / "onnx" / "model.onnx"
    tokenizer_path = model / "tokenizer.json"
    for path in (onnx_path, tokenizer_path, index):
        if not path.exists():
            raise FileNotFoundError(path)
    docs, vectors, index_meta = read_index(index)
    if not index_meta["model_key"].startswith(model.name):
        raise ValueError(
            f"인덱스의 모델({index_meta['model_key']})이 {model.name}이 아닙니다. 같은 모델로 만든 인덱스를 주세요."
        )
    codes, scales = quantize_rows(vectors)
    info = {
        "key": key,
        "label": label,
        "dim": int(vectors.shape[1]),
        "description": f"{model.name} (ONNX {onnx_path.stat().st_size / 1e6:.0f}MB, {vectors.shape[1]}차원)",
        "corpus_hash": index_meta["corpus_hash"],
    }
    tags = [
        payload_tag(f"tokenizer-{key}", tokenizer_path.read_bytes()),
        payload_tag(f"model-{key}", onnx_path.read_bytes()),
        payload_tag(f"vectors-{key}", scales.tobytes() + codes.tobytes(), compress=False),
    ]
    return info, docs, tags


@app.default
def build(
    model: Path = ROOT / "models" / "finetuned" / "r001_A-pruned-int8",
    index: Path = ROOT / "data" / "processed" / "index" / "r001_A-pruned-int8-onnx.sqlite",
    baseline_model: Path = ROOT / "models" / "multilingual-e5-small-pruned-int8",
    baseline_index: Path = ROOT / "data" / "processed" / "index" / "multilingual-e5-small-pruned-int8-onnx.sqlite",
    baseline: bool = False,
    out: Path = ROOT / "dist" / "law-search-offline.html",
    ort_dist: Path = ORT_DIST,
) -> None:
    """HTML 한 파일을 만든다. 기본은 파인튜닝 모델 하나 (--baseline이면 학습 전 모델도 넣어 나란히 비교한다).

    Args:
        model: 파인튜닝 모델 폴더 (tokenizer.json, onnx/model.onnx).
        index: 같은 모델로 만든 인덱스 파일 (ragkit index --checkpoint <model> --backend onnx).
        baseline_model: 비교할 학습 전 모델 폴더.
        baseline_index: 학습 전 모델로 만든 인덱스 파일.
        baseline: 학습 전 모델도 넣어 나란히 비교한다 (약 103MB). 기본은 파인튜닝 모델만 (약 56MB).
        out: 만들 HTML 파일.
        ort_dist: onnxruntime-web의 dist 폴더.
    """
    specs = [("finetuned", "파인튜닝", model, index)]
    if baseline:
        specs.append(("base", "학습 전", baseline_model, baseline_index))
    models, tags, docs = [], [], None
    for spec in specs:
        info, model_docs, model_tags = model_payloads(*spec)
        if docs is not None and [d["id"] for d in model_docs] != [d["id"] for d in docs]:
            raise ValueError(f"{spec[3]}: 다른 인덱스와 조문 순서가 다릅니다. 같은 코퍼스로 만든 인덱스를 주세요.")
        docs = model_docs
        models.append(info)
        tags.extend(model_tags)
    ort_js, wasm, ort_version = ort_module(ort_dist)
    meta = {
        "models": models,
        "built_at": datetime.now().astimezone().date().isoformat(),
        "docs": len(docs),
        "onnxruntime_web": ort_version,
    }

    payloads = "\n".join([
        payload_tag("meta", json.dumps(meta, ensure_ascii=False).encode()),
        payload_tag("docs", json.dumps(corpus_payload(docs), ensure_ascii=False, separators=(",", ":")).encode()),
        *tags,
        payload_tag("wasm", wasm),
    ])
    script = "\n".join([ort_js, *(inline_module(SRC / name) for name in ("tokenizer.js", "search.js", "app.js"))])
    if "</script" in script:
        raise ValueError("스크립트 안에 </script가 있어 HTML에 그대로 넣을 수 없습니다")
    html = (SRC / "template.html").read_text(encoding="utf-8")
    html = html.replace("<!-- PAYLOADS -->", payloads).replace("/* SCRIPT */", script)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("만듦: {} ({:.1f}MB, 조문 {}건, 모델 {})", out, out.stat().st_size / 1e6, len(docs),
                ", ".join(m["label"] for m in models))


if __name__ == "__main__":
    app()
