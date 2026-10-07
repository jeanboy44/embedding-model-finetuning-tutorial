"""오프라인 검색 HTML(apps/offline-search) 테스트: 벡터 양자화, HTML 조립, JS 토크나이저가 파이썬과 같은 id를 내는지."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ragkit.config import get_settings
from ragkit.retrieval import build_index

from .conftest import DOCS, fake_embed

APP_DIR = Path(__file__).parents[2] / "apps" / "offline-search"
spec = importlib.util.spec_from_file_location("offline_search_build", APP_DIR / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

PRUNED = get_settings().models_dir / "multilingual-e5-small-pruned-int8"


def test_quantize_rows_keeps_cosine_ranking() -> None:
    """행마다 배율을 둔 int8: 점수 오차가 작고 1등이 같다."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(500, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[7] + 0.1 * rng.normal(size=384).astype(np.float32)

    codes, scales = build.quantize_rows(vectors)

    assert codes.dtype == np.int8 and scales.shape == (500,)
    restored = codes.astype(np.float32) * scales[:, None]
    np.testing.assert_allclose(restored, vectors, atol=scales.max())
    assert np.abs(restored @ query - vectors @ query).max() < 0.01
    assert np.argmax(restored @ query) == np.argmax(vectors @ query) == 7


def test_read_index_and_corpus_payload_keep_rowid_order(tmp_path: Path) -> None:
    """인덱스의 조문과 벡터를 같은 순서로 읽고, 법령 정보는 법령마다 한 번만 적는다."""
    build_index(DOCS, fake_embed, tmp_path / "idx.sqlite", model_key="fake")

    docs, vectors, meta = build.read_index(tmp_path / "idx.sqlite")
    payload = build.corpus_payload(docs)

    assert [d["id"] for d in docs] == [d["id"] for d in DOCS]
    np.testing.assert_allclose(vectors, fake_embed([f"passage: {d['title']}\n{d['text']}" for d in DOCS]), atol=1e-6)
    assert meta["model_key"] == "fake"
    assert len(payload["laws"]) == len({d["law_name"] for d in DOCS})
    law_index, doc_id, parent_id, _, _ = payload["docs"][1]
    assert payload["laws"][law_index]["name"] == "근로기준법"
    assert (doc_id, parent_id) == ("제56조_제1항", "제56조")


def test_ort_module_turns_export_into_const() -> None:
    """onnxruntime-web 번들 끝의 export를 지우고 `const ort = {...}`로 감싼다."""
    if not (build.ORT_DIST / "ort.wasm.bundle.min.mjs").exists():
        pytest.skip("onnxruntime-web 미설치 (pnpm --dir apps/offline-search install)")

    js, wasm, version = build.ort_module(build.ORT_DIST)

    assert js.startswith("const ort = (() => {") and "export{" not in js
    assert "InferenceSession:" in js and wasm[:4] == b"\0asm" and version


@pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")
@pytest.mark.skipif(not (PRUNED / "tokenizer.json").exists(), reason="가지치기 모델 없음")
def test_js_tokenizer_matches_python_tokenizers() -> None:
    """브라우저용 Unigram 토크나이저가 파이썬 tokenizers와 같은 id를 낸다 (줄바꿈·전각·이모지 포함)."""
    from tokenizers import Tokenizer

    texts = [
        "query: 수습 기간에도 최저임금을 줘야 하나요?",
        "query:   여러   칸 공백  ",
        "query: ＡＢＣ①㈜ ㎏ 😀 emoji",
        "query: \t탭\r\n줄바꿈",
        "passage: 근로기준법 제55조 (휴일)\n사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장하여야 한다.",
        "passage: 여신전문금융업법 제70조 (벌칙)\n징역형과 벌금형은 병과(倂科)할 수 있다. 국제표준화기구(ISO)",
        "query: " + "아주 긴 문장 " * 400,
    ]
    tok = Tokenizer.from_file(str(PRUNED / "tokenizer.json"))
    tok.enable_truncation(max_length=512)
    expected = [e.ids for e in tok.encode_batch(texts)]

    script = f"""
        import {{ readFileSync }} from "node:fs";
        import {{ UnigramTokenizer }} from {json.dumps((APP_DIR / "src" / "tokenizer.js").as_uri())};
        const tok = new UnigramTokenizer(JSON.parse(readFileSync({json.dumps(str(PRUNED / "tokenizer.json"))}, "utf8")));
        console.log(JSON.stringify({json.dumps(texts, ensure_ascii=False)}.map((t) => tok.encode(t))));
    """
    result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)

    assert json.loads(result.stdout) == expected
