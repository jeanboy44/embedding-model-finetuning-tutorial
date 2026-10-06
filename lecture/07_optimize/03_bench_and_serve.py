"""
실습 4-3 (7교시): 원본 / ONNX / INT8 / 가지치기+INT8 비교표, 그리고 API에 적용
================================================================================

학습 목표:
- ragkit-bench로 네 변형을 같은 조건(CPU, 같은 질문)에서 잰 표를 읽는다
  지연(p50/p95) · 처리량 · 서비스 메모리 · 모델 크기 · 설치 크기 · Recall@k
- 표를 읽고 "정확도를 얼마나 잃고 무엇을 얻는가"를 DS가 직접 판단한다
- 실습 4-1 API의 백엔드를 torch → ONNX로 바꿔 시작 시간과 검색 속도를 비교한다

두 가지 모드:
- 기본: 받은 측정 결과(experiments/exp_008_deploy_bench/results/)를 읽고, API 비교만 직접 잰다 (1분 안팎)
- --run: test 질문 앞 200개로 ragkit-bench를 직접 돌린다 (수 분). 결과는 임시 폴더에 쓰고 지운다
  (받은 결과를 덮어쓰지 않는다. 전체 측정은 uv run ragkit-bench run)

사전 준비:
    Drive에서 받은 models/ (multilingual-e5-small, -int8, -pruned-int8), data/processed/index/,
    data/splits/test.jsonl, experiments/exp_008_deploy_bench/results/
    직접 만들려면 실습 4-2의 CLI(export-onnx → quantize → prune-vocab …)와 uv run ragkit index

실행:
    uv run python lecture/07_optimize/03_bench_and_serve.py
    uv run python lecture/07_optimize/03_bench_and_serve.py --run
"""

import json
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

from fastapi.testclient import TestClient
from ragkit_api.app import create_app

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "exp_008_deploy_bench" / "results" / "results.json"
TEST = ROOT / "data" / "splits" / "test.jsonl"
MODELS = ROOT / "models"
RUN_QUESTIONS = 200  # --run에서 쓸 test 질문 수
SEARCH_REPEAT = 20


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def pad(text: str, width: int) -> str:
    """화면 폭 기준으로 오른쪽을 채운다 (한글은 두 칸을 차지해 f-string 폭 지정으로는 열이 어긋난다)."""
    used = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - used, 0)


def ratio(before: float, after: float) -> str:
    """크기 변화를 '1/4.0로' · '23% 줄어듦' · '거의 같음' · '1.6배 큼'처럼 읽기 쉽게."""
    if after <= 0 or abs(before / after - 1) < 0.05:
        return "거의 같음"
    if after > before:
        return f"{after / before:.1f}배 큼"
    return (
        f"1/{before / after:.1f}로"
        if before / after >= 2
        else f"{1 - after / before:.0%} 줄어듦"
    )


def run_bench_subset(out_dir: Path) -> Path:
    """test 질문 앞부분으로 ragkit-bench를 돌려 out_dir에 결과를 쓴다."""
    lines = [line for line in TEST.read_text().splitlines() if line.strip()][
        :RUN_QUESTIONS
    ]
    subset = out_dir / "test_subset.jsonl"
    subset.write_text("\n".join(lines) + "\n")
    cmd = [
        "ragkit-bench",
        "run",
        "--questions",
        str(subset),
        "--out-dir",
        str(out_dir),
        "--n-latency",
        "30",
    ]
    print("$ " + " ".join(cmd).replace(str(out_dir), "<임시 폴더>"))
    print(
        "측정 중… (변형마다 새 프로세스, 변형별 인덱스가 없으면 먼저 만든다)",
        flush=True,
    )
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out_dir / "results.json"


# ============================================================
# 1. 측정 결과 (ragkit-bench: 변형마다 새 프로세스에서 잰다)
# ============================================================
section("1. ragkit-bench 결과")
with tempfile.TemporaryDirectory() as tmp:
    if "--run" in sys.argv:
        if not TEST.exists():
            sys.exit(f"{TEST}가 없습니다. Drive에서 data/splits/를 받으세요.")
        results = run_bench_subset(Path(tmp))
    elif RESULTS.exists():
        results = RESULTS
        print(f"받은 결과: {RESULTS.relative_to(ROOT)} (직접 재려면 --run)")
    else:
        sys.exit(
            f"{RESULTS}가 없습니다.\n  Drive에서 experiments/exp_008_deploy_bench/를 받거나 --run으로 직접 재세요."
        )
    rows = json.loads(results.read_text())["rows"]
    print((results.parent / "comparison.md").read_text())

# ============================================================
# 2. 표 읽기
# ============================================================
section("2. 무엇을 얻고 무엇을 잃었나 (원본 torch-fp32 대비)")
base = next((r for r in rows if r["variant"] == "torch-fp32"), rows[0])
for r in rows:
    if r is base:
        continue
    print(f"[{r['variant']}]")
    print(
        f"  설치 크기  {base['install_mb']:.0f} → {r['install_mb']:.0f} MB  ({ratio(base['install_mb'], r['install_mb'])})"
    )
    print(
        f"  모델 크기  {base['model_mb']:.0f} → {r['model_mb']:.0f} MB  ({ratio(base['model_mb'], r['model_mb'])})"
    )
    print(f"  로딩       {base['load_s']:.2f} → {r['load_s']:.2f} s")
    print(f"  지연 p50   {base['latency_ms_p50']:.1f} → {r['latency_ms_p50']:.1f} ms")
    print(
        f"  메모리     {base['peak_rss_mb']:.0f} → {r['peak_rss_mb']:.0f} MB  ({ratio(base['peak_rss_mb'], r['peak_rss_mb'])})"
    )
    print(
        f"  R@5        {base['recall@5']:.3f} → {r['recall@5']:.3f}  ({r['recall@5'] - base['recall@5']:+.3f})"
    )
print("""
생각해 볼 것:
- ONNX fp32는 메모리가 오히려 크다: 가중치를 모두 메모리에 올린다(torch는 safetensors를 필요할 때 읽는다)
- 처리량(문서/초)은 이 Mac에서 torch가 빠를 수 있다. 배포 서버(Linux x86)에서는 다를 수 있다 → 배포 환경에서 다시 재라
- 채널별 INT8은 Recall 손실이 거의 없이 크기·메모리·로딩이 작다
- 가지치기+INT8은 한 번 더 1/4: 한국어 법령 전용이라는 대가를 치르고 가장 작다
""")

# ============================================================
# 3. API에 적용: 모델·백엔드만 바꾼다
# ============================================================
section("3. API 백엔드 바꾸기: torch → ONNX")
for label, kwargs in [
    ("torch (실습 4-1)", {"backend": "torch"}),
    (
        "ONNX INT8",
        {"model": str(MODELS / "multilingual-e5-small-int8"), "backend": "onnx"},
    ),
    (
        "가지치기 + INT8",
        {"model": str(MODELS / "multilingual-e5-small-pruned-int8"), "backend": "onnx"},
    ),
]:
    if "model" in kwargs and not Path(kwargs["model"]).is_dir():
        print(
            f"{label}: {kwargs['model']}가 없어 건너뜁니다. Drive에서 models/를 받거나 실습 4-2의 CLI로 만드세요."
        )
        continue
    start = time.perf_counter()
    with TestClient(create_app(open_kwargs=kwargs)) as client:
        load = time.perf_counter() - start
        body = {"query": "야간 근로 수당", "k": 1, "laws": ["근로기준법"]}
        hits = client.post("/api/search", json=body).json()[
            "hits"
        ]  # 첫 요청은 워밍업으로 버린다
        times = []
        for _ in range(SEARCH_REPEAT):
            t = time.perf_counter()
            client.post("/api/search", json=body)
            times.append(1000 * (time.perf_counter() - t))
        print(
            f"{pad(label, 18)} 시작 {load:4.1f}s · 검색 중앙값 {statistics.median(times):4.0f}ms "
            f"({SEARCH_REPEAT}회) · 1위 {hits[0]['title']}"
        )
print("""
실제 서버:
  uv run --package ragkit-api ragkit-api --model models/multilingual-e5-small-pruned-int8
배포할 것: ragkit core(torch 없음) + 모델 폴더(약 30MB) + 인덱스 파일(data/processed/index/<모델 키>.sqlite)
""")
