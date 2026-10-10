"""
실습 2-3 (6교시): 실험 002 기록 보기, 에이전트가 돈 바퀴
========================================================

학습 목표:
- 02에서 적은 가설("모델이 헷갈리는 조문을 오답으로 주면 오른다")을 에이전트가 한 바퀴 돈 기록을 본다
- 바퀴의 여섯 단계가 문서 하나씩으로 남는 방식(experiments/002_hard_negatives/00~05)을 본다
- 대조군 seed 3개로 잡음 폭을 재고, 미리 적은 기준으로 판정하는 방식을 본다
- 사람이 정한 것(기준 · 제약 · 의심 · 규칙)과 에이전트가 한 것을 나눠 본다

돌리는 것은 없다. 학습 6개(run 하나 55~95분)와 오답 후보 9,009쌍 판정은 수업 시간에 할 수 없어 기록만 읽는다.
결과 모델(models/finetuned/002_hard_negatives)은 3교시에 받았고, 7 · 8교시에서 이 모델을 줄이고 제품에 넣는다.

실행:
    uv run python lecture/06_experiment/03_hard_negatives.py      # 기록 읽기 (몇 초)
"""

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "001_hard_negatives"
RUNS = RUN / "03_run"
LECTURE_TABLE = (
    ROOT / "experiments" / "exp_010_lecture_comparison" / "results" / "comparison.md"
)
LABELS = {
    "control_s42": "대조군 seed 42 (기존 LLM 오답 1개)",
    "control_s43": "대조군 seed 43",
    "control_s44": "대조군 seed 44",
    "A": "A 채굴 3개, 정답 점수 95% 이상은 버림",
    "B": "B 채굴 3개, 95% 이상은 판정해 남김",
    "C": "C 채굴 1개 (A의 1위만)",
}
DOCS = [
    (
        "00_error_analysis.md",
        "오답 분석",
        "dev 실패를 관계 6유형으로 집계, 기존 오답의 현재 순위",
    ),
    ("01_research.md", "문헌", "관련 논문의 효과 크기를 표로, 우리 분석과 맞춰 봄"),
    ("02_plan.md", "계획", "가설 · run 구성 · 통과 기준을 실행 전에 적음"),
    ("03_run/", "실행", "run 6개를 밤새 순서대로, 오답 후보 9,009쌍 판정"),
    ("04_results.md", "결과", "결과표 · 대응 검정 · 고친 질문과 망친 질문"),
    ("05_retro.md", "회고", "다음 실험 후보를 비용 · 근거와 함께 순위로"),
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_r5(model: str) -> str:
    """강의 비교표(실험 exp_010, 커밋된 comparison.md)에서 모델의 test R@5를 읽는다."""
    if not LECTURE_TABLE.exists():
        return "-"
    for line in LECTURE_TABLE.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0] == model:
            return re.sub(r"\*", "", cells[1])
    return "-"


if not (RUNS / "decision.json").exists():
    print(
        f"실험 002 기록이 없습니다: {rel(RUNS)} (저장소에 커밋되어 있다. git pull로 받는다)"
    )
    raise SystemExit(1)


# ============================================================
# 1. 바퀴의 여섯 단계 = 문서 여섯 개
# ============================================================
section("1. 바퀴의 여섯 단계가 문서로 남았다 (experiments/002_hard_negatives)")
for name, step, what in DOCS:
    print(f"  {pad(name, 22)}{pad(step, 10)}{what}")
reviews = sorted(p.stem for p in (RUN / "reviews").glob("*.md"))
print(
    f"\n  검토 문서 {len(reviews)}개 (reviews/): 검색 · 방법론 · 운영 전문가 역할 에이전트가 단계마다 검토"
)
print("  각 문서 끝의 '검토 반영' 절에 검토 의견을 어떻게 반영했는지 적었다")


# ============================================================
# 2. 계획: 실행 전에 적은 것
# ============================================================
section("2. 계획: 결과를 보기 전에 적은 것 (02_plan.md)")
print(
    "  가설   같은 법 1~2개뿐인 오답을, 모델이 직접 캔 상위권 오답 3개(다른 법 포함)로 바꾸면 오른다"
)
print(
    "  근거   1위 오답의 55%가 다른 법인데 기존 오답은 100% 같은 법, 그중 40%는 이미 50위 밖 (dev · 003 기준. 02에서 본 test · 001은 51%)"
)
print(
    "  고정   시작 모델 학습 전 e5-small · lr 3e-5 · 최대 epoch 4 · 채굴은 train 법령 문서에서만"
)
print(
    "  기준   대조군 평균 대비 Δ ≥ 0.02 이고 seed 범위보다 클 것, Holm 보정 p < 0.05 · CI 하한 > 0,"
)
print("         법령 평균이 나빠지지 않고 9개 법령 중 6개 이상이 나빠지지 않을 것")
print(
    "  의심   계획 검토에서 채굴 후보의 15~19%가 dev · test 법령 문서임을 짚음 → train 법령으로 제한"
)


# ============================================================
# 3. 실행과 결과
# ============================================================
section("3. 결과: dev 복수 정답 R@5 (1,412 질문)")
decision = read_json(RUNS / "decision.json")
print(f"  {pad('run', 40)}R@5    학습 시간  학습 행")
for run, label in LABELS.items():
    meta = read_json(RUNS / run / "train_meta.json")
    r5 = read_json(RUNS / run / "eval_dev.json")["multi"]["recall@5"]
    print(
        f"  {pad(label, 40)}{r5:.3f}  {meta['seconds'] / 60:5.0f}분   {meta['train_examples']:,}"
    )
print(
    f"\n  대조군 seed 3개: 평균 {decision['control_mean']:.3f}, 범위 {decision['control_range']:.4f}"
    " → seed만 바꿔서는 거의 흔들리지 않는다"
)
print(
    f"\n  {pad('대조군 평균 대비', 40)}Δ       95% CI              Holm p    나빠진 법령"
)
for run in ("A", "B", "C"):
    r = decision["runs"][run]
    lo, hi = r["ci95"]
    print(
        f"  {pad(LABELS[run], 40)}{r['delta_vs_control_mean']:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
        f"{r['holm_p']:.1e}  {r['laws_worse']}/9"
    )
print(
    "  → A · B · C 모두 미리 적은 기준을 넘었다. 셋 사이 차이는 잡음 범위라 판정 비용이 없는 A를 채택"
)
judge = read_json(RUNS / "judge_summary.json")
totals = judge["b_totals"]
pairs = sum(totals.values())
print(
    f"\n  B의 판정 {pairs:,}쌍: 진짜 정답 {totals['full']:,} · 부분 정답 {totals['partial']:,} · 오답 {totals['no']:,}"
    f" (다른 에이전트와의 일치 κ {judge['kappa']:.2f})"
)
print(
    "  다 판정했지만 A와 차이가 없었다 → 규칙이 하나 생겼다: '큰 판정은 300쌍 시범부터'"
)
print(
    f"\n  test R@5 (마지막에 한 번): 001 {test_r5('001_baseline')} → 002 A {test_r5('002_hard_negatives')}"
    "  (002는 lr · epoch도 001과 달라, 오답 효과는 같은 설정 대조군 대비 Δ로 본다)"
)


# ============================================================
# 4. 사람이 남는 곳
# ============================================================
section("4. 에이전트가 돌고, 사람은 고른다")
for who, what in (
    (
        "기준",
        "개선은 Δ ≥ 0.02 이고 seed 흔들림의 2배 초과일 때. test는 마지막에 한 번만",
    ),
    (
        "제약",
        "모델은 e5-small 하나, 학습할 때도 다른 모델은 쓰지 않는다. reranker 없음",
    ),
    ("의심", "채굴 후보의 dev · test 누수, test를 보고 고르지 않았나"),
    (
        "규칙",
        "큰 판정은 300쌍 시범부터 (9,009쌍을 다 판정했는데 효과가 없었던 실수에서)",
    ),
):
    print(f"  {pad(who, 6)}{what}")
print(
    "\n  02에서 여러분이 한 일과 같다: 신뢰구간으로 판정하고, 새로 틀린 질문을 의심하고,"
)
print(
    "  닿지 않는 범위(31위 밖)를 적고, 다음 가설을 골랐다. 달라진 것은 그 사이를 누가 채우느냐다"
)
print(f"\n원문: {rel(RUN)}/04_results.md · 05_retro.md")
