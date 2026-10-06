"""
실습 4-2 (7교시): ONNX 변환, INT8 양자화, 어휘 가지치기
========================================================

학습 목표:
- PyTorch 모델과 ONNX 모델의 결과가 같은지(코사인 유사도) 직접 확인한다
- 동적 INT8 양자화로 모델 파일이 1/4로 줄어드는 것을 본다
- 양자화 방식(텐서 단위 vs 채널별)에 따라 정확도 손실이 다름을 잰다 (--run)
  → 크기는 같아도 "어떻게 줄이느냐"가 정확도를 좌우한다
- 용량의 대부분이 어휘 임베딩 표임을 확인하고, 어휘 가지치기로 한 번 더 줄인다

두 가지 모드:
- 기본: 받은 모델 폴더 네 개를 비교만 한다 (변환·양자화를 다시 하지 않는다, 약 20초)
      models/multilingual-e5-small              원본 (+ onnx/model.onnx)
      models/multilingual-e5-small-int8         ONNX INT8 (채널별)
      models/multilingual-e5-small-pruned       어휘 가지치기 (+ onnx/model.onnx)
      models/multilingual-e5-small-pruned-int8  가지치기 + INT8
- --run: 임시 폴더에 직접 변환 → 양자화(텐서 단위·채널별) → 가지치기 → 변환 → 양자화 (약 30초)
      임시 폴더는 끝나면 지운다. 받은 모델 폴더는 건드리지 않는다

사전 준비:
    Drive에서 받은 models/ (또는 아래 CLI로 직접 만들기)
    --run에는 extra [train]이 필요: uv sync --all-packages --all-extras

실행:
    uv run python lecture/07_optimize/02_onnx_quantize_prune.py
    uv run python lecture/07_optimize/02_onnx_quantize_prune.py --run

같은 작업을 CLI로:
    uv run ragkit export-onnx models/multilingual-e5-small
    uv run ragkit quantize models/multilingual-e5-small [--no-per-channel]
    uv run ragkit prune-vocab models/multilingual-e5-small  (→ export-onnx → quantize)
"""

import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

from ragkit.config import get_settings
from ragkit.data import doc_text, load_corpus
from ragkit.embeddings import create_embedding_fn, format_passages, format_queries

SETTINGS = get_settings()
MODELS = SETTINGS.models_dir
MODEL = MODELS / "multilingual-e5-small"
RECEIVED = {
    "fp32": MODEL,
    "int8": MODELS / "multilingual-e5-small-int8",
    "pruned": MODELS / "multilingual-e5-small-pruned",
    "pruned-int8": MODELS / "multilingual-e5-small-pruned-int8",
}
BENCH = SETTINGS.experiments_dir / "exp_008_deploy_bench" / "results" / "results.json"
GRANULARITY = SETTINGS.experiments_dir / "exp_012_int8_granularity" / "results" / "comparison.md"
TEST = SETTINGS.data_dir / "splits" / "test.jsonl"
SAMPLE = 200
RUN = "--run" in sys.argv


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def pad(text: str, width: int) -> str:
    """화면 폭 기준으로 오른쪽을 채운다 (한글은 두 칸을 차지해 f-string 폭 지정으로는 열이 어긋난다)."""
    used = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - used, 0)


def onnx_file(model_dir: Path) -> Path:
    return model_dir / "onnx" / "model.onnx"


def cosine_to(ref: np.ndarray, model_dir: Path, texts: list[str]) -> np.ndarray:
    """model_dir(onnx)로 만든 임베딩과 기준 임베딩의 행별 코사인 (둘 다 정규화되어 있다)."""
    vec = create_embedding_fn(str(model_dir), backend="onnx")(texts)
    return (vec * ref).sum(axis=1)


# ============================================================
# 0. 준비물 확인
# ============================================================
need = [MODEL / "model.safetensors", TEST] + (
    [] if RUN else [onnx_file(p) for p in RECEIVED.values()]
)
missing = [p for p in need if not p.exists()]
if missing:
    print("다음 파일이 없습니다:")
    for p in missing:
        print(f"  {p}")
    sys.exit(
        "Drive에서 models/ · data/splits/를 받거나, --run으로 임시 폴더에 직접 만들어 보세요."
    )

docs = load_corpus(SETTINGS.data_dir / "processed" / "law_docs.json")
texts = format_passages([doc_text(d) for d in docs[:: len(docs) // SAMPLE][:SAMPLE]])
texts += format_queries(
    ["편의점 알바 주휴수당", "전세 보증금 돌려받기", "음주운전 벌금"]
)
questions = [
    json.loads(line)["query"] for line in TEST.read_text().splitlines() if line.strip()
]

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    # --run이면 임시 폴더에 직접 만든 것을, 아니면 받은 폴더를 쓴다
    paths = dict(RECEIVED)

    # ============================================================
    # 1. PyTorch → ONNX
    # ============================================================
    section("1. ONNX 변환 (torch 없이 돌릴 수 있는 형식)")
    if RUN:
        from ragkit.models.onnx_export import export_onnx, quantize_onnx
        from ragkit.models.vocab_prune import prune_vocab, vocab_keep_ids

        print("변환 중… (임시 폴더)")
        export_onnx(MODEL, work / "fp32")
        paths["fp32"] = work / "fp32"
    fp32_onnx = onnx_file(paths["fp32"])
    print(
        f"onnx/model.onnx {mb(fp32_onnx):.0f} MB (가중치 model.safetensors {mb(MODEL / 'model.safetensors'):.0f} MB와 거의 같다)"
    )

    torch_vec = create_embedding_fn(str(MODEL), backend="torch", device="cpu")(texts)
    onnx_vec = create_embedding_fn(str(paths["fp32"]), backend="onnx")(texts)
    cos = (torch_vec * onnx_vec).sum(axis=1)
    print(
        f"torch vs onnx 코사인 ({len(texts)}문장): 평균 {cos.mean():.5f}, 최소 {cos.min():.5f} → 사실상 같은 모델"
    )

    # ============================================================
    # 2. INT8 양자화
    # ============================================================
    section("2. 동적 INT8 양자화 (가중치 float32 4바이트 → int8 1바이트)")
    rows = [("fp32 원본", mb(fp32_onnx), 1.0, 1.0)]
    if RUN:
        for name, key, per_channel in [
            ("INT8 텐서 단위", "int8-tensor", False),
            ("INT8 채널별", "int8", True),
        ]:
            print(f"{name} 양자화 중…")
            quantize_onnx(paths["fp32"], work / key, per_channel=per_channel)
            paths[key] = work / key
            c = cosine_to(onnx_vec, paths[key], texts)
            rows.append((name, mb(onnx_file(paths[key])), c.mean(), c.min()))
    else:
        c = cosine_to(onnx_vec, paths["int8"], texts)
        rows.append(("INT8 채널별", mb(onnx_file(paths["int8"])), c.mean(), c.min()))

    print(
        f"\n{pad('방식', 16)}{pad('크기 MB', 9)}{pad('fp32 대비 코사인 평균', 24)}최소"
    )
    for name, size, mean, low in rows:
        print(f"{pad(name, 16)}{size:<9.0f}{mean:<24.4f}{low:.4f}")

    print(
        """
왜 방식에 따라 다른가:
- 텐서 단위: 가중치 행렬 전체에 스케일 하나. 값 범위가 넓은 열 때문에 나머지 열의 정밀도가 뭉개진다
- 채널별: 출력 채널(열)마다 스케일을 따로 둔다. 크기는 거의 같고 원본에 훨씬 가깝다
  → ragkit quantize는 채널별이 기본값이다"""
        + ("" if RUN else " (텐서 단위와 직접 비교하려면 --run)")
    )
    if BENCH.exists():
        bench = {r["variant"]: r for r in json.loads(BENCH.read_text())["rows"]}
        n = next(iter(bench.values()))["n_questions"]
        print(
            f"\n검색 정확도 R@5 (test 질문 {n:,}개, {BENCH.relative_to(SETTINGS.project_root)}):"
        )
        for variant in ("onnx-fp32", "onnx-int8", "onnx-int8-pruned"):
            if variant in bench:
                print(f"  {variant:<18} {bench[variant]['recall@5']:.3f}")
    if GRANULARITY.exists():
        # 실험 012: 같은 test로 텐서 단위 INT8까지 잰 비교표 (comparison.md의 R@5 열)
        r5 = {}
        for line in GRANULARITY.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) > 2 and cells[1].startswith("**"):
                r5[cells[0]] = float(cells[1].strip("*"))
        if "multilingual-e5-small-int8-tensor" in r5:
            print(f"\n텐서 단위 vs 채널별 R@5 ({GRANULARITY.relative_to(SETTINGS.project_root)}):")
            print(f"  fp32 {r5.get('intfloat/multilingual-e5-small', float('nan')):.3f} · "
                  f"채널별 {r5.get('multilingual-e5-small-int8', float('nan')):.3f} · "
                  f"텐서 단위 {r5['multilingual-e5-small-int8-tensor']:.3f}  → 코사인 0.99의 작은 차이가 검색에서는 R@5 몇 점으로 커진다")  # fmt: skip

    # ============================================================
    # 3. 어디를 줄여야 효과가 큰가: 어휘 가지치기
    # ============================================================
    section("3. 용량은 어디에 있나 → 어휘 가지치기")
    with safe_open(str(MODEL / "model.safetensors"), "np") as f:
        sizes = {k: int(np.prod(f.get_slice(k).get_shape())) for k in f.keys()}  # noqa: SIM118 (safetensors API, dict 아님)
    emb = sum(v for k, v in sizes.items() if "word_embeddings" in k)
    total = sum(sizes.values())
    print(
        f"전체 {total / 1e6:.1f}M 파라미터 중 단어 임베딩 표 {emb / 1e6:.1f}M ({emb / total:.0%})"
    )
    print(
        "→ 다국어 모델이라 100개 언어의 어휘를 모두 들고 있다. 층을 줄이기 전에 어휘부터 보자"
    )

    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    corpus_texts = format_passages([doc_text(d) for d in docs])
    used: set[int] = set()
    for i in range(0, len(corpus_texts), 2000):
        for enc in tok.encode_batch(corpus_texts[i : i + 2000]):
            used.update(enc.ids)
    vocab = tok.get_vocab_size()
    print(
        f"법령 코퍼스가 실제로 쓰는 토큰: {len(used):,}개 / {vocab:,}개 ({len(used) / vocab:.1%})"
    )

    if RUN:
        # 코퍼스 토큰 + 질문 앞 문구 + 한글 조각 전체 + 짧은 ASCII(약어·숫자·기호) + 특수 토큰
        prefixes = [SETTINGS.query_prefix, SETTINGS.passage_prefix]
        keep = vocab_keep_ids(tok, corpus_texts, prefixes=prefixes, ascii_max_len=3)
        print("가지치기 → 변환 → 양자화 중… (임시 폴더)")
        paths["pruned"] = prune_vocab(MODEL, work / "pruned", keep)
        export_onnx(paths["pruned"])
        quantize_onnx(paths["pruned"], work / "pruned-int8")
        paths["pruned-int8"] = work / "pruned-int8"

    new_tok = Tokenizer.from_file(str(paths["pruned"] / "tokenizer.json"))
    formatted = format_queries(questions)
    same = sum(new_tok.encode(q).tokens == tok.encode(q).tokens for q in formatted)
    c = cosine_to(onnx_vec, paths["pruned-int8"], texts)
    int8_mb = next(size for name, size, _, _ in rows if name == "INT8 채널별")

    print(f"\n어휘 {vocab:,} → {new_tok.get_vocab_size():,}개")
    print(f"  {pad('fp32 원본', 18)}{mb(fp32_onnx):4.0f} MB")
    print(f"  {pad('INT8 (채널별)', 18)}{int8_mb:4.0f} MB")
    print(
        f"  {pad('가지치기 + INT8', 18)}{mb(onnx_file(paths['pruned-int8'])):4.0f} MB  fp32 대비 코사인 평균 {c.mean():.4f}"
    )
    print(
        f"\ntest 질문 {len(questions):,}개 중 {same:,}개({same / len(questions):.1%})가 원본과 똑같이 토큰화"
    )
    if same == len(questions):
        print("→ 이 질문들에서는 가지치기로 바뀌는 것이 없다")
        print(
            "  (위 코사인이 1보다 조금 작은 것은 가지치기가 아니라 INT8 양자화 때문이다)"
        )
    else:
        print(
            f"→ {len(questions) - same:,}개는 남기지 않은 토큰이 있어 더 잘게 쪼개졌다. 이 질문들은 결과가 달라질 수 있다"
        )

print(f"""
정리:
- 양자화는 모든 가중치를 근사한다(정밀도를 줄인다). 가지치기는 쓰지 않는 행을 뺄 뿐, 남은 가중치는 그대로다
- 그래서 남긴 어휘로 덮이는 입력은 원본과 결과가 같다. 대가는 범용성: 한국어 법령 전용 모델이 된다
  (남기지 않은 문자는 더 잘게 쪼개지거나 <unk>. 실제 사용자 질문으로 덮는 비율을 꼭 확인할 것)
- CLI: uv run ragkit prune-vocab models/multilingual-e5-small → export-onnx → quantize
{"- (--run) 임시 폴더에 만든 모델은 지웠다. 받은 models/는 그대로다" + chr(10) if RUN else ""}
다음(실습 4-3): 원본 / ONNX / INT8 / 가지치기+INT8을 속도·메모리·크기·정확도로 한 표에 비교하고, API에 적용한다.
""")

# torch와 onnxruntime을 한 프로세스에서 쓰면 macOS에서 종료 처리 중 가끔
# "recursive_mutex lock failed"로 죽는다(출력은 이미 끝난 뒤). 정리 단계를 건너뛰고 끝낸다.
sys.stdout.flush()
os._exit(0)
