"""
3단계-3: 원본 / ONNX / INT8 / 가지치기+INT8 비교표 만들기, 그리고 API에 적용
=============================================================================

학습 목표:
- ragkit-bench로 네 변형을 같은 조건(CPU, 같은 질문)에서 잰다
  지연(p50/p95) · 처리량 · 서비스 메모리 · 모델 크기 · 설치 크기 · Recall@k
- 표를 읽고 "정확도를 얼마나 잃고 무엇을 얻는가"를 DS가 직접 판단한다
- 2단계 API의 백엔드를 torch → ONNX로 바꿔 시작 시간과 검색 속도를 비교한다

사전 준비:
    uv run ragkit index
    uv run ragkit split data/questions          # data/splits/test.jsonl
    (02를 실행했거나 아래 모델 폴더가 있어야 한다)
    uv run ragkit quantize models/multilingual-e5-small
    uv run ragkit prune-vocab models/multilingual-e5-small
    uv run ragkit export-onnx models/multilingual-e5-small-pruned
    uv run ragkit quantize models/multilingual-e5-small-pruned

실행:
    uv run python tutorials/03_optimize/03_bench_and_deploy.py            # 저장된 결과가 있으면 재사용
    uv run python tutorials/03_optimize/03_bench_and_deploy.py --rerun    # 다시 측정 (수 분)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from ragkit_api.app import create_app

RESULTS = Path("experiments/exp_008_deploy_bench/results/results.json")


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 1. 측정 (ragkit-bench: 변형마다 새 프로세스)
# ============================================================
section("1. ragkit-bench run")
if "--rerun" in sys.argv or not RESULTS.exists():
    print("측정 중… (변형별 인덱스가 없으면 먼저 만든다)")
    subprocess.run(["ragkit-bench", "run"], check=True)
rows = json.loads(RESULTS.read_text())["rows"]
print((RESULTS.parent / "comparison.md").read_text())

# ============================================================
# 2. 표 읽기
# ============================================================
section("2. 무엇을 얻고 무엇을 잃었나 (원본 torch-fp32 대비)")
base = next(r for r in rows if r["variant"] == "torch-fp32")
for r in rows[1:]:
    print(f"[{r['variant']}]")
    print(f"  설치 크기  {base['install_mb']:.0f} → {r['install_mb']:.0f} MB  (x{base['install_mb'] / r['install_mb']:.1f} 작음)")
    print(f"  모델 크기  {base['model_mb']:.0f} → {r['model_mb']:.0f} MB  (x{base['model_mb'] / r['model_mb']:.1f} 작음)")
    print(f"  로딩       {base['load_s']:.2f} → {r['load_s']:.2f} s")
    print(f"  지연 p50   {base['latency_ms_p50']:.1f} → {r['latency_ms_p50']:.1f} ms")
    print(f"  메모리     {base['peak_rss_mb']:.0f} → {r['peak_rss_mb']:.0f} MB")
    print(f"  R@5        {base['recall@5']:.3f} → {r['recall@5']:.3f}  ({r['recall@5'] - base['recall@5']:+.3f})")
print("""
생각해 볼 것:
- ONNX fp32는 메모리가 오히려 크다: 가중치를 모두 메모리에 올린다(torch는 safetensors를 필요할 때 읽는다)
- 처리량(문서/초)은 이 Mac에서 torch가 빠르다. 배포 서버(Linux x86)에서는 다를 수 있다 → 배포 환경에서 다시 재라
- 채널별 INT8은 Recall 손실이 거의 없이 크기·메모리·로딩이 작다
- 가지치기+INT8은 한 번 더 1/4: 한국어 법령 전용이라는 대가를 치르고 가장 작다
""")

# ============================================================
# 3. API에 적용: 모델·백엔드만 바꾼다
# ============================================================
section("3. API 백엔드 바꾸기: torch → ONNX")
for label, kwargs in [
    ("torch (2단계)", {"backend": "torch"}),
    ("ONNX INT8", {"model": "models/multilingual-e5-small-int8", "backend": "onnx"}),
    ("가지치기 + INT8", {"model": "models/multilingual-e5-small-pruned-int8", "backend": "onnx"}),
]:
    start = time.perf_counter()
    with TestClient(create_app(open_kwargs=kwargs)) as client:
        load = time.perf_counter() - start
        t = time.perf_counter()
        hits = client.post("/api/search", json={"query": "야간 근로 수당", "k": 1, "laws": ["근로기준법"]}).json()["hits"]
        print(f"{label:16s} 시작 {load:4.1f}s · 검색 {1000 * (time.perf_counter() - t):4.0f}ms · 1위 {hits[0]['title']}")
print("""
실제 서버:
  uv run --package ragkit-api ragkit-api --model models/multilingual-e5-small-pruned-int8
배포할 것: ragkit core(torch 없음) + 모델 폴더(약 30MB) + 인덱스 파일(data/processed/index/<모델 키>.sqlite)
""")
