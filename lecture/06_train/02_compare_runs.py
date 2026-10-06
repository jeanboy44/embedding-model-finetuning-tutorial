"""
실습 3-2 (6교시): 실험 비교, 가설은 맞았나
==========================================

학습 목표:
- 실험 한 바퀴(가설 → 설정 → 파일럿 → 학습 → dev로 선택 → test 1회 → 분석 → 다음 가설)를 실험 010 비교표로 따라간다
- 학습 전 / 002 전체 학습 / 004 LoRA / 006 배치 128 / 001 다시 캔 오답 3개를 같은 test로 비교한다
- 두 모델의 차이가 우연인지 질문 단위 대응 검정(ragkit paired-test)으로 확인한다
- 5교시의 쿼리 확장 행을 옆에 두고 "파인튜닝은 질문당 LLM 비용을 학습 1회로 옮긴다"를 확인한다

사전 준비 (없으면 스크립트가 받는 명령을 알려 주고 끝난다):
    uv run python scripts/data_version.py pull v1     # 코퍼스 + 분할
    models/multilingual-e5-small, models/finetuned/{exp_002,exp_004,exp_006,r001_A},
    data/processed/index/*.sqlite                      # 강사 Drive (인덱스가 있으면 질문만 임베딩한다)

실행:
    uv run python lecture/06_train/02_compare_runs.py          # test 전체 비교 (처음 약 3분, 이후 결과 재사용)
    uv run python lecture/06_train/02_compare_runs.py --run    # test 300개 표본으로 비교와 검정을 직접 (1분 안쪽)

기본 모드 결과는 experiments/results/lecture/06_train/exp_010/ 에 쓴다(03_error_analysis.py가 이어서 읽는다).
강사가 실험 010 결과 JSON(experiments/exp_010_lecture_comparison/results/*.json)을 나눠 줬으면 그것을 읽는다.
받은 비교표(comparison.md)는 덮어쓰지 않는다.
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

from ragkit.config import get_settings
from ragkit.data import load_questions
from ragkit.retrieval import default_index_path, model_key

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
TEST = DATA / "splits" / "test.jsonl"
EXPERIMENT = ROOT / "experiments" / "exp_010_lecture_comparison"
CONFIG = EXPERIMENT / "config.yaml"
RECEIVED = EXPERIMENT / "results"  # 강사가 나눠 준 결과 JSON이 있으면 여기
OUT = ROOT / "experiments" / "results" / "lecture" / "06_train" / "exp_010"
EXPANSION = (
    ROOT
    / "experiments"
    / "results"
    / "lecture"
    / "05_evaluate"
    / "query_expansion"
    / "comparison.json"
)
MODELS = ROOT / "models"
# 모델 폴더 이름 → (표에 쓸 이름, 가설). 실험 설정은 experiments/exp_00X, runs/001_hard_negatives
RUNS = {
    "multilingual-e5-small": ("학습 전 e5-small", "출발점 (5교시)"),
    "exp_002": ("002 전체 학습", "기준 실험: 학습하면 오른다"),
    "exp_004": ("004 LoRA", "파라미터 0.4%로도 대부분 따라간다"),
    "exp_006": ("006 배치 128", "in-batch 오답이 많을수록 오른다"),
    "r001_A": ("001 다시 캔 오답 3개", "모델이 헷갈리는 조문을 오답으로 주면 오른다"),
}
RUN_SAMPLE = 300
SEED = 42

GET_DATA = "uv run python scripts/data_version.py pull v1"
FROM_DRIVE = "강사 Drive에서 받아 {path}에 둔다"


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
        print(f"  - {rel(path)}\n      받기: {how.format(path=rel(path))}")
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


def load_results(folder: Path) -> dict[str, tuple[dict, Path]]:
    """결과 폴더의 모델별 JSON을 모델 폴더 이름 → (결과, 경로)로 읽는다 (comparison.json 제외)."""
    found = {}
    for path in sorted(folder.glob("*.json")):
        if path.name == "comparison.json" or path.stem.endswith("_expand"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if "per_question" in result:
            found[Path(result["model"]).name] = (result, path)
    return found


def law_mean(result: dict) -> float:
    """법령별 R@5의 단순 평균 (질문이 많은 법령에 끌려가지 않게 본다)."""
    laws = result.get("by_law") or {}
    return (
        sum(g["doc"]["recall@5"] for g in laws.values()) / len(laws)
        if laws
        else result["doc"]["recall@5"]
    )


def train_meta(name: str) -> dict | None:
    path = MODELS / "finetuned" / name / "train_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
parser = argparse.ArgumentParser(description="실습 3-2: 실험 비교")
parser.add_argument(
    "--run",
    action="store_true",
    help=f"test {RUN_SAMPLE}개 표본으로 비교와 검정을 직접 실행",
)
args = parser.parse_args()

require(
    [
        (CORPUS, GET_DATA),
        (TEST, GET_DATA),
        (MODELS / "multilingual-e5-small", FROM_DRIVE),
        *[
            (MODELS / "finetuned" / name, FROM_DRIVE)
            for name in RUNS
            if name != "multilingual-e5-small"
        ],
    ]
)


# ============================================================
# 1. 실험 한 바퀴
# ============================================================
section("1. 실험 한 바퀴 = 가설 하나")
print(
    "  가설 → 설정(config.yaml) → 파일럿(--max-steps) → 학습 → dev로 epoch·설정 고르기 → test 1회 → 분석 → 다음 가설"
)
print(
    "  test는 고르는 데 쓰지 않는다. 그래야 '처음 보는 법령에서의 성능'이라고 말할 수 있다\n"
)
for name, (label, hypothesis) in RUNS.items():
    print(f"  {pad(label, 22)}{hypothesis}")
print(
    "\n  001의 가설은 002의 오답 분석에서 나왔다 (runs/001_hard_negatives/00_error_analysis.md)"
)


# ============================================================
# 2. 결과 모으기: ragkit compare (실험 010)
# ============================================================
if args.run:
    section(f"2. --run: test {RUN_SAMPLE}개 표본으로 ragkit compare")
    tmp = Path(tempfile.mkdtemp(prefix="lecture-03-2-"))
    sample = random.Random(SEED).sample(load_questions(TEST), RUN_SAMPLE)
    subset = tmp / "test_sample.jsonl"
    subset.write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in sample),
        encoding="utf-8",
    )
    out_dir = tmp / "results"
    seconds = ragkit(
        "compare", str(CONFIG), "--questions", str(subset), "--out-dir", str(out_dir)
    )
    print(f"  ({seconds:.0f}초, 표본 {RUN_SAMPLE}개 · seed {SEED})")
    results = load_results(out_dir)
else:
    section("2. 결과 모으기: ragkit compare (실험 010, test 전체)")
    results = load_results(RECEIVED)
    out_dir = RECEIVED
    if set(RUNS) <= set(results):
        print(f"강사가 나눠 준 결과를 읽는다: {rel(RECEIVED)}")
    else:
        results = load_results(OUT)
        out_dir = OUT
        if set(RUNS) <= set(results):
            print(
                f"이전에 실행한 결과를 다시 읽는다: {rel(OUT)} (새로 재려면 이 폴더를 지운다)"
            )
        else:
            for name in RUNS:
                folder = (
                    MODELS / name
                    if name == "multilingual-e5-small"
                    else MODELS / "finetuned" / name
                )
                index = default_index_path(
                    model_key(
                        name if name == "multilingual-e5-small" else str(folder),
                        None if name == "multilingual-e5-small" else folder,
                    )
                )
                if not index.exists():
                    print(f"참고: {rel(index)}가 없어 새로 만든다 (모델마다 7~13분)")
            print(
                "모델 5개를 test 2,526개로 평가한다. 인덱스가 있으면 질문만 임베딩한다."
            )
            seconds = ragkit("compare", str(CONFIG), "--out-dir", str(OUT))
            print(f"  ({seconds / 60:.1f}분)")
            results = load_results(OUT)
missing = [name for name in RUNS if name not in results]
if missing:
    print(
        f"결과가 없는 모델: {', '.join(missing)}. 위 ragkit compare 출력을 확인하세요."
    )
    raise SystemExit(1)


# ============================================================
# 3. 비교표
# ============================================================
n = results["multilingual-e5-small"][0]["n"]
section(f"3. 비교표 (test {n:,}개, 처음 보는 법령 16개, 코퍼스 전체에서 검색)")
print(
    "  "
    + pad("모델", 24)
    + "R@5    R@1    R@10   MRR@10 법령평균  학습 시간  학습 파라미터"
)
for name, (label, _) in RUNS.items():
    r = results[name][0]
    d = r["doc"]
    meta = train_meta(name)
    cost = (
        f"{meta['seconds'] / 60:5.0f}분     {meta['trainable_params'] / meta['total_params']:6.1%}"
        if meta
        else "    -          -"
    )
    print(
        f"  {pad(label, 24)}{d['recall@5']:.3f}  {d['recall@1']:.3f}  {d['recall@10']:.3f}  "
        f"{d['mrr@10']:.3f}  {law_mean(r):.3f}   {cost}"
    )

print("\n질문 유형별 R@5")
types = list(results["multilingual-e5-small"][0]["by_query_type"])
print("  " + pad("모델", 24) + "".join(pad(t, 11) for t in types))
for name, (label, _) in RUNS.items():
    groups = results[name][0]["by_query_type"]
    print(
        "  "
        + pad(label, 24)
        + "".join(pad(f"{groups[t]['doc']['recall@5']:.3f}", 11) for t in types)
    )

print("\n테마별 R@5")
themes = list(results["multilingual-e5-small"][0]["by_theme"])
print("  " + pad("모델", 24) + "".join(pad(t, 10) for t in themes))
for name, (label, _) in RUNS.items():
    groups = results[name][0]["by_theme"]
    print(
        "  "
        + pad(label, 24)
        + "".join(pad(f"{groups[t]['doc']['recall@5']:.3f}", 10) for t in themes)
    )


# ============================================================
# 4. 가설은 맞았나
# ============================================================
section("4. 가설은 맞았나 (숫자는 위 표에서, 0.02 미만 차이는 차이로 보지 않는다)")
r5 = {name: results[name][0]["doc"]["recall@5"] for name in RUNS}
base, full, lora, big, mined = (r5[k] for k in RUNS)
lora_meta, full_meta = train_meta("exp_004"), train_meta("exp_002")
MIN_DIFF = (
    0.02  # 실험 기록의 기준 (runs/README.md): 이보다 작은 차이는 의미 없음으로 본다
)


def verdict(delta: float) -> str:
    if delta >= MIN_DIFF:
        return "올랐다"
    return "내렸다" if delta <= -MIN_DIFF else "차이 없음"


print(
    f"  파인튜닝     학습 전 {base:.3f} → 002 {full:.3f} ({full - base:+.3f}, {verdict(full - base)})."
    " 처음 보는 법령에서 잰 값이다"
)
if lora_meta and full_meta:
    share = lora_meta["trainable_params"] / lora_meta["total_params"]
    print(
        f"  LoRA         파라미터 {share:.1%}만 학습해 002의 {lora / full:.0%} ({lora:.3f} 대 {full:.3f})"
        + (". 가설대로 대부분 따라간다" if lora / full >= 0.9 else "")
    )
print(
    f"  배치 128     002 대비 {big - full:+.3f} ({verdict(big - full)})."
    + (
        " 오답 수를 4배로 늘려도 오르지 않았다. 가설이 틀렸다"
        if big - full < MIN_DIFF
        else ""
    )
)
print(
    "               늘어난 in-batch 오답은 대부분 다른 법령의 쉬운 문서다 (docs/PLAN.md 학습 실험 결과)"
)
print(
    f"  다시 캔 오답 002 대비 {mined - full:+.3f} ({verdict(mined - full)})."
    + (" 다만 001은 lr 3e-5 · 4 epoch로 학습 설정도 다르다" if mined - full >= MIN_DIFF else "")
)
print(
    "               같은 설정의 대조군끼리 보면 dev 복수 정답 R@5 0.720 → 0.761 (+0.041, runs/001_hard_negatives/04_results.md)."
)
print("               설정을 맞춰도 오르므로, 필요한 것은 오답의 수가 아니라 난이도다")
if args.run:
    print(
        f"  (표본 {RUN_SAMPLE}개라 전체 test와 숫자가 다르다. 우연인지는 아래 검정으로 본다)"
    )


# ============================================================
# 5. 차이가 우연인가: 질문 단위 대응 검정
# ============================================================
section("5. 차이가 우연인가: ragkit paired-test (같은 질문끼리 짝지어 비교)")
base_path = results["exp_002"][1]
for other in ("exp_006", "r001_A"):
    print(f"\n002 → {RUNS[other][0]}")
    ragkit("paired-test", str(base_path), str(results[other][1]), "--judge", "doc")
print(
    "\n  읽는 법: 95% CI가 0을 걸치면 차이를 주장하지 않는다. 고침/망침은 한쪽만 맞힌 질문 수"
)
if args.run:
    print(
        f"  표본 {RUN_SAMPLE}개라 신뢰구간이 전체 test보다 넓다. 같은 차이도 표본이 작으면 확신할 수 없다"
    )


# ============================================================
# 6. 쿼리 확장과 나란히: 비용을 어디서 내나
# ============================================================
section("6. 학습 없는 선택지와 나란히 (5교시 표 + 오늘 표)")
print(
    "  "
    + pad("선택지", 24)
    + pad("test R@5", 36)
    + pad("질문당 LLM 호출", 18)
    + "한 번 드는 비용"
)
print("  " + pad("학습 전 e5-small", 24) + pad(f"{base:.3f}", 36) + pad("0", 18) + "-")
expansion_note = "측정 전 (확장 캐시가 test 일부뿐)"
if EXPANSION.exists():
    table = json.loads(EXPANSION.read_text(encoding="utf-8"))
    row = next((m for m in table["models"] if m.get("expansion")), None)
    plain = next((m for m in table["models"] if not m.get("expansion")), None)
    if row and plain:
        expansion_note = (
            f"측정 전 ({table['n']}개 표본: {plain['doc']['recall@5']:.3f}"
            f"→{row['doc']['recall@5']:.3f})"
        )
print(
    "  "
    + pad("e5-small + 쿼리 확장", 24)
    + pad(expansion_note, 36)
    + pad("1 (매 질문)", 18)
    + "-"
)
mined_meta = train_meta("r001_A")
train_cost = (
    f"학습 {mined_meta['seconds'] / 60:.0f}분 1회" if mined_meta else "학습 1회"
)
print(
    "  " + pad("001 파인튜닝", 24) + pad(f"{mined:.3f}", 36) + pad("0", 18) + train_cost
)
print(
    "\n  → 파인튜닝은 질문마다 내던 LLM 비용·지연을 학습 1회로 옮긴다. 모델 크기(118M, 384차원)도 그대로다"
)


# ============================================================
# 7. 기록: MLflow
# ============================================================
section("7. 실험 기록 (MLflow)")
uri = get_settings().mlflow_tracking_uri
if uri:
    print(
        f"MLFLOW_TRACKING_URI={uri} → 이번 ragkit compare·train이 run으로 기록됐다. 브라우저에서 run을 나란히 비교한다"
    )
else:
    print(
        ".env에 MLFLOW_TRACKING_URI를 넣으면 ragkit train·evaluate·compare가 코드 변경 없이 설정·지표·결과를 기록한다"
    )
    print(
        "  uv run --extra mlflow mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db \\"
    )
    print("      --artifacts-destination mlruns/artifacts --port 5050")
    print(
        "  .env: MLFLOW_TRACKING_URI=http://127.0.0.1:5050   (macOS는 5000번을 AirPlay가 쓴다)"
    )
print(f"\n결과 폴더: {rel(out_dir)} (모델별 JSON의 per_question = 질문별 순위·top-10)")
print("다음: 03_error_analysis.py — 무엇을 고쳤고 무엇이 여전히 틀리나")
