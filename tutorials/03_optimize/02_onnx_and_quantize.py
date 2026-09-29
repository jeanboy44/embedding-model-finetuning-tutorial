"""
3단계-2: ONNX 변환과 INT8 양자화
================================

학습 목표:
- PyTorch 모델을 ONNX로 변환하고, 결과가 원본과 같은지(코사인 유사도) 직접 확인한다
- 동적 INT8 양자화로 모델 파일을 1/4로 줄인다
- 양자화 방식(텐서 단위 vs 채널별)에 따라 정확도 손실이 크게 다름을 잰다
  → 크기는 같아도 "어떻게 줄이느냐"가 정확도를 좌우한다

사전 준비:
    uv run python scripts/download_model_hf.py     # models/multilingual-e5-small
    (uv sync --all-packages --all-extras: 변환·양자화에는 extra [train]이 필요)

실행:
    uv run python tutorials/03_optimize/02_onnx_and_quantize.py

같은 작업을 CLI로:
    uv run ragkit export-onnx models/multilingual-e5-small
    uv run ragkit quantize models/multilingual-e5-small [--no-per-channel]
"""

import tempfile
from pathlib import Path

import numpy as np

from ragkit.config import get_settings
from ragkit.data import doc_text, load_corpus
from ragkit.embeddings import create_embedding_fn
from ragkit.models.onnx_export import export_onnx, quantize_onnx

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
- 실측(전체 코퍼스, 질문 443개): 텐서 단위 R@5 0.451 / 채널별 0.535 = fp32 0.535
  → ragkit quantize는 채널별이 기본값이다

다음(03): 원본 / ONNX / INT8을 속도·메모리·설치 크기·정확도로 한 표에 비교하고, API에 적용한다.
""")
