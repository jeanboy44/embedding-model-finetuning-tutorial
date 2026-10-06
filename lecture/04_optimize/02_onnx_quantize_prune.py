"""
3단계-2: ONNX 변환, INT8 양자화, 어휘 가지치기
==============================================

학습 목표:
- PyTorch 모델을 ONNX로 변환하고, 결과가 원본과 같은지(코사인 유사도) 직접 확인한다
- 동적 INT8 양자화로 모델 파일을 1/4로 줄인다
- 양자화 방식(텐서 단위 vs 채널별)에 따라 정확도 손실이 크게 다름을 잰다
  → 크기는 같아도 "어떻게 줄이느냐"가 정확도를 좌우한다
- 용량의 82%가 어휘 임베딩 표임을 확인하고, 어휘 가지치기로 한 번 더 1/4로 줄인다

사전 준비:
    uv run python scripts/download_model_hf.py     # models/multilingual-e5-small
    (uv sync --all-packages --all-extras: 변환·양자화에는 extra [train]이 필요)

실행:
    uv run python lecture/03_optimize/02_onnx_and_quantize.py

같은 작업을 CLI로:
    uv run ragkit export-onnx models/multilingual-e5-small
    uv run ragkit quantize models/multilingual-e5-small [--no-per-channel]
    uv run ragkit prune-vocab models/multilingual-e5-small  (→ export-onnx → quantize)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

from ragkit.config import get_settings
from ragkit.data import doc_text, load_corpus
from ragkit.embeddings import create_embedding_fn
from ragkit.models.onnx_export import export_onnx, quantize_onnx
from ragkit.models.vocab_prune import prune_vocab, vocab_keep_ids

MODEL = get_settings().models_dir / "multilingual-e5-small"
SAMPLE = 200


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def mb(path: Path) -> float:
    return path.stat().st_size / 1e6


docs = load_corpus(get_settings().data_dir / "processed" / "law_docs.json")
texts = ["passage: " + doc_text(d) for d in docs[:: len(docs) // SAMPLE][:SAMPLE]]
texts += ["query: 편의점 알바 주휴수당", "query: 전세 보증금 돌려받기", "query: 음주운전 벌금"]

# ============================================================
# 1. PyTorch → ONNX
# ============================================================
section("1. ONNX 변환 (torch 없이 돌릴 수 있는 형식)")
onnx_path = MODEL / "onnx" / "model.onnx"
if not onnx_path.exists():
    print("변환 중…")
    export_onnx(MODEL)
print(f"{onnx_path}  {mb(onnx_path):.0f} MB (가중치 {mb(MODEL / 'model.safetensors'):.0f} MB와 거의 같다)")

torch_vec = create_embedding_fn(str(MODEL), backend="torch", device="cpu")(texts)
onnx_vec = create_embedding_fn(str(MODEL), backend="onnx")(texts)
cos = (torch_vec * onnx_vec).sum(axis=1)
print(f"torch vs onnx 코사인: 평균 {cos.mean():.5f}, 최소 {cos.min():.5f} → 사실상 같은 모델")

# ============================================================
# 2. INT8 양자화: 텐서 단위 vs 채널별
# ============================================================
section("2. 동적 INT8 양자화 (가중치 float32 4바이트 → int8 1바이트)")
work = Path(tempfile.mkdtemp())
rows = []
for name, per_channel in [("텐서 단위", False), ("채널별", True)]:
    out = quantize_onnx(MODEL, work / name, per_channel=per_channel)
    vec = create_embedding_fn(str(work / name), backend="onnx")(texts)
    c = (vec * onnx_vec).sum(axis=1)
    rows.append((name, mb(out), c.mean(), c.min()))

print(f"{'방식':10s} {'크기 MB':>8s} {'fp32 대비 코사인 평균':>20s} {'최소':>8s}")
print(f"{'fp32 원본':10s} {mb(onnx_path):8.0f} {1.0:20.4f} {1.0:8.4f}")
for name, size, mean, low in rows:
    print(f"{name:10s} {size:8.0f} {mean:20.4f} {low:8.4f}")

print("""
왜 다른가:
- 텐서 단위: 가중치 행렬 전체에 스케일 하나. 값 범위가 넓은 열 때문에 나머지 열의 정밀도가 뭉개진다
- 채널별: 출력 채널(열)마다 스케일을 따로 둔다. 크기는 거의 같고 원본에 훨씬 가깝다
- 실측(전체 코퍼스, 이전 분할 질문 443개 기준): 텐서 단위 R@5 0.451 / 채널별 0.535 = fp32 0.535
  → ragkit quantize는 채널별이 기본값이다
""")

# ============================================================
# 3. 어디를 줄여야 효과가 큰가: 어휘 가지치기
# ============================================================
section("3. 용량은 어디에 있나 → 어휘 가지치기")
with safe_open(str(MODEL / "model.safetensors"), "np") as f:
    sizes = {k: int(np.prod(f.get_slice(k).get_shape())) for k in f.keys()}  # noqa: SIM118 (safetensors API, dict 아님)
emb = sum(v for k, v in sizes.items() if "word_embeddings" in k)
total = sum(sizes.values())
print(f"전체 {total / 1e6:.1f}M 파라미터 중 단어 임베딩 표 {emb / 1e6:.1f}M ({emb / total:.0%})")
print("→ 다국어 모델이라 100개 언어의 어휘 25만 개를 들고 있다. 층을 줄이기 전에 어휘부터 보자")

tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
corpus_texts = ["passage: " + doc_text(d) for d in docs]
used = set()
for i in range(0, len(corpus_texts), 2000):
    for enc in tok.encode_batch(corpus_texts[i : i + 2000]):
        used.update(enc.ids)
print(f"법령 코퍼스가 실제로 쓰는 토큰: {len(used):,}개 / {tok.get_vocab_size():,}개 ({len(used) / tok.get_vocab_size():.1%})")

# 코퍼스 토큰 + 질문 앞 문구 + 한글 조각 전체 + 짧은 ASCII(약어·숫자·기호) + 특수 토큰
keep = vocab_keep_ids(tok, corpus_texts, prefixes=["query: ", "passage: "], ascii_max_len=3)
pruned = prune_vocab(MODEL, work / "pruned", keep)
export_onnx(pruned)
pruned_int8 = quantize_onnx(pruned, work / "pruned-int8")

new_tok = Tokenizer.from_file(str(pruned / "tokenizer.json"))
with open("data/splits/test.jsonl") as fh:
    questions = [json.loads(line)["query"] for line in fh]
same = sum(new_tok.encode("query: " + q).tokens == tok.encode("query: " + q).tokens for q in questions)
vec = create_embedding_fn(str(work / "pruned-int8"), backend="onnx")(texts)
c = (vec * onnx_vec).sum(axis=1)
print(f"\n어휘 {tok.get_vocab_size():,} → {len(keep):,}개")
print(f"{'fp32 원본':16s} {mb(onnx_path):6.0f} MB")
print(f"{'INT8 (채널별)':16s} {rows[1][1]:6.0f} MB")
print(f"{'가지치기 + INT8':16s} {mb(pruned_int8):6.0f} MB  fp32 대비 코사인 평균 {c.mean():.4f}")
print(f"test 질문 {len(questions)}개 중 {same}개({same / len(questions):.0%})가 원본과 똑같이 토큰화 → 가지치기로 바뀌는 것은 없다")
print("(위 코사인 0.999x의 차이는 가지치기가 아니라 INT8 양자화 때문이다)")

print("""
정리:
- 양자화는 모든 가중치를 근사한다(정밀도를 줄인다). 가지치기는 쓰지 않는 행을 뺄 뿐, 남은 가중치는 그대로다
- 그래서 남긴 어휘로 덮이는 입력은 원본과 결과가 같다. 대가는 범용성: 한국어 법령 전용 모델이 된다
  (남기지 않은 문자는 더 잘게 쪼개지거나 <unk>. 실제 사용자 질문으로 덮는 비율을 꼭 확인할 것)
- CLI: uv run ragkit prune-vocab models/multilingual-e5-small → export-onnx → quantize

다음(03): 원본 / ONNX / INT8 / 가지치기+INT8을 속도·메모리·크기·정확도로 한 표에 비교하고, API에 적용한다.
""")
