"""
3단계-1: 임베딩(인덱싱) 속도 옵션 비교
=======================================

학습 목표:
- 같은 모델이라도 실행 방법에 따라 인덱싱 시간이 크게 달라짐을 직접 잰다
- 옵션별 효과를 분리해서 본다
  1) 백엔드: torch vs ONNX
  2) 장치: CPU vs Apple GPU(mps) / CUDA
  3) 길이순 배치: 배치는 가장 긴 문장에 맞춰 패딩되므로, 길이가 비슷한 것끼리 묶으면 계산이 준다
  4) 스레드 수
- 속도를 바꿔도 결과(임베딩)는 같아야 한다 → 기준 대비 코사인 유사도로 확인

실행:
    uv run python tutorials/03_optimize/01_embedding_speed.py

직접 바꿔 볼 것:
    SAMPLE_SIZE, BATCH_SIZE, CONFIGS (아래 상수)
    같은 옵션은 인덱싱 명령에도 있다:
    uv run ragkit index --backend torch --device mps --batch-size 64 --threads 8
    .env로 기본값을 바꿀 수도 있다: EMBEDDING_DEVICE, EMBEDDING_SORT_BY_LENGTH, EMBEDDING_NUM_THREADS
"""

import time

import numpy as np
import onnxruntime as ort

from ragkit.config import get_settings
from ragkit.data import doc_text, load_corpus
from ragkit.embeddings import create_embedding_fn, format_passages

SAMPLE_SIZE = 1000  # 코퍼스에서 고르게 뽑을 문서 수
BATCH_SIZE = 64

# (이름, backend, device, sort_by_length, threads, onnx_providers)
CONFIGS = [
    ("torch  cpu  정렬 없음 (기준)", "torch", "cpu", False, None, None),
    ("torch  cpu  길이 정렬", "torch", "cpu", True, None, None),
    ("torch  cpu  길이 정렬, 4스레드", "torch", "cpu", True, 4, None),
    ("torch  mps  정렬 없음", "torch", "mps", False, None, None),
    ("torch  mps  길이 정렬", "torch", "mps", True, None, None),
    ("onnx   cpu  정렬 없음", "onnx", None, False, None, None),
    ("onnx   cpu  길이 정렬", "onnx", None, True, None, None),
    ("onnx   coreml 길이 정렬", "onnx", None, True, None, ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
]


def available(device: str | None, providers: list[str] | None) -> bool:
    """이 컴퓨터에서 쓸 수 있는 설정인지 확인한다."""
    if providers and providers[0] not in ort.get_available_providers():
        return False
    if device in ("mps", "cuda"):
        import torch

        return torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    return True


settings = get_settings()
docs = load_corpus(settings.data_dir / "processed" / "law_docs.json")
step = max(len(docs) // SAMPLE_SIZE, 1)
texts = format_passages([doc_text(d) for d in docs[::step][:SAMPLE_SIZE]])
lengths = np.array([len(t) for t in texts])
print(f"문서 {len(texts):,}개 (전체 {len(docs):,}개에서 고르게), 길이 중앙값 {int(np.median(lengths))}자, "
      f"최대 {lengths.max():,}자, 배치 {BATCH_SIZE}")

baseline = None
rows = []
for name, backend, device, sort, threads, providers in CONFIGS:
    if not available(device, providers):
        print(f"{name:32s}  (이 컴퓨터에서 사용 불가, 건너뜀)")
        continue
    try:
        embed = create_embedding_fn(
            settings.embedding_model_name,
            backend=backend,
            device=device,
            sort_by_length=sort,
            num_threads=threads,
            onnx_providers=providers,
        )
        embed(texts[:BATCH_SIZE], batch_size=BATCH_SIZE)  # 워밍업 (모델 로딩·첫 실행 비용 제외)
        start = time.perf_counter()
        vectors = embed(texts, batch_size=BATCH_SIZE)
        elapsed = time.perf_counter() - start
    except Exception as e:  # noqa: BLE001 - 설정별로 실패 이유만 보여 주고 다음 설정으로 넘어간다
        print(f"{name:32s}  실패: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
        continue
    if baseline is None:
        baseline = vectors
    same = float((vectors * baseline).sum(axis=1).min())
    full_min = elapsed / len(texts) * len(docs) / 60
    rows.append((name, elapsed, full_min, same))
    print(f"{name:32s}  {elapsed:6.1f}초  전체 약 {full_min:5.1f}분  기준 대비 최소 코사인 {same:.4f}")

base_time = rows[0][1]
print("\n기준(torch cpu, 정렬 없음) 대비 속도")
for name, elapsed, _, _ in rows:
    print(f"  {name:32s}  x{base_time / elapsed:.1f}")

print("""
생각해 볼 것:
- 가장 효과가 큰 옵션은 무엇인가? 길이 정렬은 왜 빨라지는가?
- 코사인이 0.999 아래로 떨어진 설정이 있다면, 속도와 정확도 중 무엇을 택할까?
- 배포 서버(Linux, GPU 없음)에서는 결과가 어떻게 달라질까? 여기서 잰 숫자는 이 컴퓨터 기준이다.
""")
