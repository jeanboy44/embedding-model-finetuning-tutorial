"""
실습 2-1 (6교시): 가설 표와 001 판정, 정말 올랐나
==================================================

학습 목표:
- 기본 실험 001 위에 세운 가설(004 LoRA · 005 배치 128 · 003 수렴 · 002 오답 다시 캐기 · model soup)과 결과를 한 표로 본다
- 학습 전 e5-small과 001을 같은 test로 재고, 차이가 우연인지 질문 단위 대응 검정(ragkit paired-test)으로 판정한다
- 평균이 올라도 새로 틀린 질문이 있다는 것을 숫자로 본다

학습은 하지 않는다. 001은 5교시에 받은 모델(models/finetuned/001_baseline)을 쓴다.

사전 준비 (없으면 스크립트가 받는 명령을 알려 주고 끝난다):
    uv run python scripts/data_version.py pull v1          # 코퍼스 + 분할
    uv run python scripts/finetuned_drive.py download 001_baseline   # 001 모델 + 인덱스 (5교시에 받음)

실행:
    uv run python lecture/06_experiment/01_compare_runs.py          # test 2,526개로 판정 (처음 약 3분, 이후 결과 재사용)
    uv run python lecture/06_experiment/01_compare_runs.py --run    # test 300개 표본으로 평가와 검정을 직접 (1분 안쪽)

기본 모드 결과는 experiments/results/lecture/06_experiment/exp_010/ 에 쓴다(02_error_analysis.py가 이어서 읽는다).
강사가 실험 exp_010 결과 JSON(experiments/archive/exp_010_lecture_comparison/results/*.json)을 나눠 줬으면 그것을 읽는다.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

from ragkit.data import load_questions

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
TEST = DATA / "splits" / "test.jsonl"
EXPERIMENT = ROOT / "experiments" / "exp_010_lecture_comparison"
RECEIVED = EXPERIMENT / "results"  # 강사가 나눠 준 결과 JSON이 있으면 여기
OUT = ROOT / "experiments" / "results" / "lecture" / "06_experiment" / "exp_010"
MODELS = ROOT / "models"
BASE_NAME, TUNED_NAME = "multilingual-e5-small", "001_baseline"
BASE_MODEL = "intfloat/multilingual-e5-small"
TUNED_DIR = MODELS / "finetuned" / TUNED_NAME
RUN_SAMPLE = 300
SEED = 42

GET_DATA = "uv run python scripts/data_version.py pull v1"
GET_002 = "uv run python scripts/finetuned_drive.py download 001_baseline"

# 우리가 세운 가설. 결과는 실험 기록에서 옮겼다 (test: experiments/archive/exp_010_lecture_comparison/results/comparison.md,
# dev: experiments/README.md · experiments/002_hard_negatives/04_results.md). model soup은 설계만 하고 돌리지 않았다
HYPOTHESES = [
    (
        "001 기본 실험",
        "우리 데이터로 학습하면 오른다",
        "test 0.669 (학습 전 0.513)",
        "오늘 직접 판정",
    ),
    (
        "004 LoRA",
        "가중치 0.4%만 학습해도 001과 같다",
        "test 0.641, Δ −0.028 CI [−0.040, −0.017]",
        "틀림",
    ),
    (
        "005 배치 128",
        "배치 안 오답을 4배로 늘리면 오른다",
        "test 0.659, Δ −0.010 CI [−0.019, +0.000]",
        "틀림",
    ),
    (
        "003 수렴 확인",
        "덜 학습됐다, epoch를 늘리면 오른다",
        "dev 0.712 → 0.729 (+0.017), 이후 과적합",
        "틀림",
    ),
    (
        "002 오답 다시 캐기",
        "모델이 헷갈리는 조문을 오답으로 주면 오른다",
        "dev 대조군 0.720 → 0.761, test 0.723",
        "맞음 (03에서 기록 보기)",
    ),
    ("model soup", "여러 번 학습한 가중치를 평균하면 오른다", "-", "설계만 하고 뺐다"),
]
SOUP_REASONS = [
    "재료 하나가 학습 1회라, 재료 수만큼 학습 시간이 든다",
    "결과는 같은 e5-small 하나라 작아지지도 빨라지지도 않는다 (배포 쪽 이득 없음)",
    "작은 모델 하나를 한 번 학습해 배포한다는 이 과제의 방향과 맞지 않는다",
]


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def pad(text: str, width: int) -> str:
    """한글(전각)을 두 칸으로 세어 왼쪽 정렬한다."""
    used = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - used, 1)


def require(items: list[tuple[Path, str]]) -> None:
    """필요한 파일이 없으면 무엇이 없고 어떻게 받는지 알려 주고 끝낸다."""
    missing = [(path, how) for path, how in items if not path.exists()]
    if not missing:
        return
    print("필요한 산출물이 없습니다.")
    for path, how in missing:
        print(f"  - {rel(path)}\n      받기: {how}")
    print(
        "준비 상태는 uv run python lecture/03_setup/01_doctor.py 로 한 번에 볼 수 있습니다."
    )
    raise SystemExit(1)


def ragkit(*args: str) -> float:
    """ragkit CLI를 저장소 루트에서 실행하고 걸린 시간(초)을 돌려준다."""
    shown = [rel(Path(a)) if a.startswith(str(ROOT)) else a for a in args]
    print(f"$ uv run ragkit {' '.join(shown)}")
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    start = time.perf_counter()
    done = subprocess.run(
        [str(Path(sys.executable).parent / "ragkit"), *args],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if done.returncode != 0:
        print(
            f"ragkit {args[0]}이 실패했습니다 (종료 코드 {done.returncode}). 위 메시지를 확인하세요."
        )
        raise SystemExit(done.returncode)
    return time.perf_counter() - start


def load_results(folder: Path) -> dict[str, Path]:
    """결과 폴더의 모델별 JSON을 모델 폴더 이름 → 경로로 찾는다 (comparison.json 제외)."""
    found = {}
    for path in sorted(folder.glob("*.json")):
        if path.name == "comparison.json" or path.stem.endswith("_expand"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if "per_question" in result:
            found[Path(result["model"]).name] = path
    return found


def evaluate_both(questions: Path | None, out_dir: Path) -> dict[str, Path]:
    """학습 전과 001을 같은 질문으로 평가해 결과 JSON 경로를 돌려준다. 인덱스가 있으면 질문만 임베딩한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, model in ((BASE_NAME, BASE_MODEL), (TUNED_NAME, str(TUNED_DIR))):
        out = out_dir / f"{name}.json"
        extra = ["--splits", str(questions.parent)] if questions else []
        seconds = ragkit(
            "evaluate", model, "--split", "test", *extra, "--out", str(out)
        )
        print(f"  ({seconds:.0f}초)\n")
        paths[name] = out
    return paths


sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
parser = argparse.ArgumentParser(description="실습 2-1: 가설 표와 001 판정")
parser.add_argument(
    "--run",
    action="store_true",
    help=f"test {RUN_SAMPLE}개 표본으로 평가와 검정을 직접 실행",
)
args = parser.parse_args()

require([(CORPUS, GET_DATA), (TEST, GET_DATA), (TUNED_DIR, GET_002)])


# ============================================================
# 1. 우리가 세운 가설
# ============================================================
section("1. 우리가 세운 가설: 001 위에 하나씩 바꿔 쌓았다")
for name, hypothesis, result, verdict in HYPOTHESES:
    print(f"  {pad(name, 20)}{hypothesis}")
    print(f"  {pad('', 20)}→ {result} · {verdict}")
print("\n  model soup을 뺀 이유 (돌리기 전에 비용과 방향을 따졌다)")
for reason in SOUP_REASONS:
    print(f"    · {reason}")
print(
    "\n  판정 규칙: 0.02 미만 차이는 차이로 보지 않는다 (dev 표준오차 약 ±0.012, experiments/README.md)"
)
print(
    "  오늘은 001을 직접 판정하고 쪼개 본다. 002의 가설이 어디서 나왔는지는 02에서 찾는다"
)


# ============================================================
# 2. 학습 전 vs 001을 같은 test로
# ============================================================
if args.run:
    section(f"2. --run: test {RUN_SAMPLE}개 표본으로 학습 전 vs 001")
    tmp = Path(tempfile.mkdtemp(prefix="lecture-06-1-"))
    sample = random.Random(SEED).sample(load_questions(TEST), RUN_SAMPLE)
    subset = tmp / "splits" / "test.jsonl"
    subset.parent.mkdir()
    subset.write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in sample),
        encoding="utf-8",
    )
    paths = evaluate_both(subset, tmp / "results")
    out_dir = tmp / "results"
else:
    section("2. 학습 전 vs 001 (test 2,526개, 처음 보는 법령 16개)")
    paths, out_dir = {}, None
    for folder in (RECEIVED, OUT):
        found = load_results(folder)
        if BASE_NAME in found and TUNED_NAME in found:
            paths, out_dir = found, folder
            print(
                f"이미 있는 결과를 읽는다: {rel(folder)} (새로 재려면 {rel(OUT)}를 지운다)"
            )
            break
    if not paths:
        print(
            "두 모델을 test 2,526개로 평가한다. 받은 인덱스가 있으면 질문만 임베딩한다."
        )
        paths = evaluate_both(None, OUT)
        out_dir = OUT

base = json.loads(paths[BASE_NAME].read_text(encoding="utf-8"))
tuned = json.loads(paths[TUNED_NAME].read_text(encoding="utf-8"))
print(f"\n  {pad('', 18)}R@1    R@5    R@10   MRR@10")
for label, r in (("학습 전 e5-small", base), ("001 전체 학습", tuned)):
    d = r["doc"]
    print(
        f"  {pad(label, 18)}{d['recall@1']:.3f}  {d['recall@5']:.3f}  {d['recall@10']:.3f}  {d['mrr@10']:.3f}"
    )
print(
    f"\n  평균만 보면 {base['doc']['recall@5']:.3f} → {tuned['doc']['recall@5']:.3f}. 이게 우연이 아닌지는 질문 단위로 본다"
)


# ============================================================
# 3. 차이가 우연인가: 질문 단위 대응 검정
# ============================================================
section("3. 차이가 우연인가: ragkit paired-test (같은 질문끼리 짝지어 비교)")
ragkit("paired-test", str(paths[BASE_NAME]), str(paths[TUNED_NAME]), "--judge", "doc")
print("\n  읽는 법")
print("    · 95% CI가 0을 걸치지 않으면 차이가 있다고 말한다")
print(
    "    · 고침 = 학습 전엔 틀리고 001이 맞힌 질문, 망침 = 그 반대. 평균이 올라도 망친 질문이 있다"
)
print("    · 나빠진 법령 수: 특정 법령에서만 오른 것인지 본다")
if args.run:
    print(
        f"  표본 {RUN_SAMPLE}개라 신뢰구간이 전체 test보다 넓다. 같은 차이도 표본이 작으면 확신할 수 없다"
    )

print(f"\n결과 폴더: {rel(out_dir)} (모델별 JSON의 per_question = 질문별 순위·top-10)")
print(
    "다음: 02_error_analysis.py — 어디서 올랐고, 무엇이 여전히 틀리고, 우리가 준 오답은 어땠나"
)
