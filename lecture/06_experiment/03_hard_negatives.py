"""
실습 2-3 (6교시): 실험 002 기록 보기, 에이전트가 돈 바퀴
========================================================

학습 목표:
- 02에서 적은 가설("모델이 헷갈리는 조문을 오답으로 주면 오른다")을 에이전트가 한 바퀴 돈 기록을 본다
- 바퀴의 여섯 단계가 문서 하나씩으로 남는 방식(experiments/002_hard_negatives/00~05)을 본다
- 공용 대조군 seed 3개로 잡음 폭을 재고, 미리 적은 기준 7개로 판정하는 방식을 본다
- 사람이 정한 것(기준 · 제약 · 의심 · 규칙)과 에이전트가 한 것을 나눠 본다

돌리는 것은 없다. 학습 run 4개(파일럿 1개 + 본 run 3개, run 하나 38~70분)와 train 9,126개 채굴은
수업 시간에 할 수 없어 기록만 읽는다.
결과 모델(models/finetuned/002_hard_negatives)은 3교시에 받았고, 7 · 8교시에서 이 모델을 줄이고 제품에 넣는다.

무엇을 하나:
    1. experiments/002_hard_negatives/의 문서 여섯 단계와 검토 문서 수를 나열한다 (저장소에 있는 것만)
    2. 02_plan.md에 실행 전에 적은 가설 · 근거 · 고정 · 기준 · 의심 · 탐색을 요약해 출력한다
    3. 04_results.md의 결과표(공용 대조군 3 seed · A1 s42/s43 · A3)를 읽어 출력하고, 판정 기준 7개 통과를 보여 준다
    4. 05_retro.md의 규칙과 함께, 사람이 정한 기준 · 제약 · 의심 · 규칙을 정리한다

실행:
    uv run python lecture/06_experiment/03_hard_negatives.py      # 기록 읽기 (몇 초)
"""

from pathlib import Path

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console(emoji=False)  # 데이터 문장의 :...: 이 이모지로 바뀌지 않게

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments" / "002_hard_negatives"  # 지금 002 (조 단위 분할, 2026-10-11)
RESULTS_MD = RUN / "04_results.md"
RUNS = (
    RUN / "03_run"
)  # 학습 로그 · config 사본. 저장소에는 없을 수 있다(결과 JSON은 커밋하지 않는다)
DOCS = [
    (
        "00_error_analysis.md",
        "오답 분석",
        "dev-a 실패 181개를 관계 6유형으로 집계, 기존 오답의 001 순위, 라벨 표본 40개 판정",
    ),
    (
        "01_research.md",
        "문헌",
        "hard negative 채굴 논문의 효과 크기를 표로, 우리 분석과 맞춰 봄",
    ),
    (
        "02_plan.md",
        "계획",
        "가설 · 고정 · run 구성 · 누수 점검 · 통과 기준을 실행 전에 적음",
    ),
    ("03_run/", "실행", "파일럿 1개 + 본 run 3개(A1 s42 · s43, A3 s42)를 순서대로"),
    (
        "04_results.md",
        "결과",
        "결과표 · 판정 기준 7개 검증 · 고친 질문과 망친 질문 · 배포 형태 dev",
    ),
    (
        "05_retro.md",
        "회고",
        "잘된 것 · 아쉬운 것 · 다음 바퀴 규칙 · 다음 가설 후보 순위",
    ),
]
# 04_results.md 결과표에서 읽을 행 (첫 칸 이름 → 출력 라벨)
TABLE_ROWS = {
    "공용 대조군": "공용 대조군 seed 42 · 43 · 44 (기존 LLM 오답 1개)",
    "A1_s42": "A1 seed 42 (001이 캔 오답 1개)",
    "A1_s43": "A1 seed 43 (seed 확인)",
    "A1 (평균)": "A1 평균 (채택 후보)",
    "A3_s42": "A3 seed 42 (캔 오답 3개, 탐색 전용)",
}


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold]{escape(title)}[/]", style="blue")


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def results_table(path: Path) -> dict[str, list[str]]:
    """04_results.md의 첫 결과표를 읽어 run 이름 → 칸 목록으로 돌려준다 (굵게 표시 **는 뗀다)."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [
            c.strip().replace("**", "") for c in line.strip().strip("|").split("|")
        ]
        if cells and cells[0] in TABLE_ROWS and cells[0] not in rows:
            rows[cells[0]] = cells
    return rows


if not RESULTS_MD.exists():
    console.print(
        Panel(
            f"실험 002 기록이 없습니다: {escape(rel(RESULTS_MD))} (저장소에 커밋되어 있다. [dim]git pull[/]로 받는다)",
            title="필요한 산출물이 없습니다",
            border_style="red",
        )
    )
    raise SystemExit(1)


# ============================================================
# 1. 바퀴의 여섯 단계 = 문서 여섯 개
# ============================================================
# 실험 폴더에 실제로 있는 문서만 "있음"으로 표시한다. 03_run/은 학습 로그라 커밋하지 않아 없을 수 있다.
# reviews/는 검색 · 방법론 전문가 역할 에이전트가 계획을 검토한 원문이다
section("1. 바퀴의 여섯 단계가 문서로 남았다 (experiments/002_hard_negatives)")
table = Table("문서", "단계", "내용", box=box.SIMPLE_HEAD, show_edge=False)
for name, step, what in DOCS:
    exists = (RUN / name.rstrip("/")).exists()
    table.add_row(
        f"[dim]{escape(name)}[/]",
        step,
        escape(what) + ("" if exists else "  [yellow](이 저장소엔 없음)[/]"),
    )
console.print(table)
reviews = sorted(p.stem for p in (RUN / "reviews").glob("*.md"))
console.print(
    f"  검토 문서 {len(reviews)}개 (reviews/: {escape(', '.join(reviews))}): 검색 · 방법론 전문가 역할 에이전트가 계획을 검토"
)
console.print("  02_plan.md 끝의 '검토 반영' 표에 지적 하나하나를 어떻게 반영했는지 적었다")


# ============================================================
# 2. 계획: 실행 전에 적은 것
# ============================================================
# 02_plan.md의 가설 · 근거 · 고정 · 기준 · 의심을 한 줄씩 옮겼다. 숫자는 00_error_analysis.md(dev-a · 001 기준)와
# 계획의 시험 채굴에서 나왔다. 02에서 본 test · 001 기준 그림(다른 법이 1위인 실패가 많다)과 같다
section("2. 계획: 결과를 보기 전에 적은 것 (02_plan.md)")
console.print(
    "  [bold]가설[/]   학습 설정과 질문 8,689행은 그대로 두고, LLM이 같은 법에서 고른 오답 1개를 001이 직접 캔 오답 1개로 바꾸면 오른다"
)
console.print(
    "  [bold]근거[/]   dev-a 실패 1위 오답의 37%가 다른 법, 32%가 법률↔시행령인데 기존 오답은 다른 법 0% · 같은 조 조각 50%,"
)
console.print(
    "         기존 오답의 39%는 이미 50위 밖, 실패 1위가 그 질문의 오답인 경우 3.3% (00 · dev-a · 001 기준)"
)
console.print(
    "  [bold]고정[/]   시작 모델 학습 전 e5-small · 001과 같은 설정(배치 32 · lr 2e-5 · 3 epoch) · 채굴은 001로, 후보는 train 조 문서만"
)
console.print("         정답 점수의 95% 이상인 후보는 가짜 오답 의심으로 버린다 (판정 없음)")
console.print(
    "  [bold]기준[/]   공용 대조군 seed 3개 평균 대비 Δ ≥ 0.02 이고 seed 범위보다 클 것, 대응 검정 CI 하한 > 0 · Holm p < 0.05,"
)
console.print(
    "         나빠진 법령 0개, dev-b도 오를 것, seed 2개가 각각 대조군 최댓값보다 높을 것 (결과를 보기 전에 고정)"
)
console.print(
    "  [bold]의심[/]   검토에서 채굴 후보의 19.6%가 dev · test 정답 조 문서임을 짚음 → 후보 풀에서 뺌(누수 점검 10항목)"
)
console.print(
    "         95% 컷이 정답보다 위에 있는 후보를 모두 버려, 학습 질문의 45.7%는 '2등과 간격 벌리기' 신호만 받는다고 미리 적음"
)
console.print(
    "  [bold]탐색[/]   오답 3개(A3)는 seed 1개만 돌려 보고만 한다. 조건을 넘어도 채택하지 않는다 (승자의 저주 방지)"
)


# ============================================================
# 3. 실행과 결과
# ============================================================
# 04_results.md의 결과표를 읽어 run별 dev R@5 · Δ · CI · 나빠진 법령을 출력한다. 모두 dev(1,243개) 숫자이고
# test는 쓰지 않았다. 03_run/이 있으면 그 안의 로그가 원본이다
section("3. 결과: dev doc R@5 (1,243 질문, 04_results.md)")
table = results_table(RESULTS_MD)
result_table = Table(
    "run", "R@5", "Δ", "95% CI", "Holm p", "나빠진 법령", "학습", box=box.SIMPLE_HEAD, show_edge=False
)
for column in result_table.columns[1:6]:
    column.justify = "right"
for key, label in TABLE_ROWS.items():
    cells = table.get(key)
    if cells is None:
        result_table.add_row(escape(label), "[dim](표에 없음)[/]")
        continue
    r5, delta, p, ci, worse, took = (
        cells[4],
        cells[5],
        cells[6],
        cells[7],
        cells[10],
        cells[11],
    )
    r5 = r5.split(" ")[0]  # 대조군 칸은 "0.7144 (0.7128~0.7168)"
    result_table.add_row(
        escape(label),
        f"[bold]{escape(r5)}[/]",
        escape(delta),
        escape(ci),
        escape(p),
        escape(worse),
        escape(took),
    )
console.print(result_table)
control = table.get("공용 대조군", [""] * 12)[4]
console.print(
    f"  공용 대조군 seed 3개: {escape(control)} → seed만 바꿔서는 0.004 범위 안에서만 흔들린다"
)
console.print("  판정 기준 7개 (04_results.md '판정 기준 검증'): A1이 모두 통과")
for rule in (
    "Δ(평균) +0.0390 ≥ 0.02",
    "Δ > 대조군 seed 범위 0.0040",
    "대응 검정 95% CI 하한 > 0 (s42 +0.0209, s43 +0.0201)",
    "McNemar p(Holm) < 0.05 (9.1e-06)",
    "법령 평균 0.7498 → 0.7836, 유의하게 나빠진 법령 0개",
    "dev-b Δ +0.0376 (동반 상승)",
    "seed 2개 재현 (0.7538 · 0.7530, 편차 0.0008)",
):
    console.print(f"    · {escape(rule)}")
console.print(
    "  고친 질문 76 · 새로 틀린 질문 30 (A1 s42 vs 대조군 s42). 02에서 본 001 판정과 같은 방식이다"
)
console.print(
    "  → [bold]A1 채택[/]. A3(오답 3개)는 0.7611로 더 높지만 학습 시간 1.8배 · seed 1개라 계획대로 채택하지 않고 다음 가설 근거로 남겼다"
)
console.print(
    "  → 다음 바퀴 003이 A3를 seed 2개로 다시 재서 Δ +0.0052, 기준 미달로 기각했다 (01의 가설 표)"
)
console.print(
    "\n  배포 형태(어휘 가지치기 + INT8, 29.7MB)의 dev R@5 0.7546: 줄여도 성능이 그대로다 (7교시)"
)
console.print("  test는 아직 쓰지 않았다 (dev로만 골랐다). 001만 test로 한 번 판정했다")
if RUNS.exists():
    console.print(f"  학습 로그 · 설정 원본: [dim]{escape(rel(RUNS))}[/]")


# ============================================================
# 4. 사람이 남는 곳
# ============================================================
# 02_plan.md의 기준 · 제약 · 의심과 05_retro.md의 규칙을 사람이 정한 것으로 모았다.
# 02에서 수강생이 직접 한 판정 · 의심 · 가설 고르기와 같은 일이다
section("4. 에이전트가 돌고, 사람은 고른다")
for who, what in (
    (
        "기준",
        "개선은 Δ ≥ 0.02 이고 seed 범위(SD의 2배)를 넘을 때. 채택 후보는 A1 하나, test는 쓰지 않는다",
    ),
    (
        "제약",
        "모델은 e5-small 하나, 학습할 때도 다른 모델은 쓰지 않는다. reranker · BM25 없음. 새 재료는 train 조에서만",
    ),
    (
        "의심",
        "채굴 후보의 dev · test 누수, 채굴 모델이 대조군 s42와 같아 s42 대비 검정이 유리하지 않나 (s43 · s44 대비도 보고)",
    ),
    (
        "규칙",
        "다음 바퀴의 대조군은 채택한 A1이다. 무거운 설정은 30 step 파일럿부터. 점수가 낮다고 다시 돌리지 않는다 (05_retro · 02_plan)",
    ),
):
    console.print(f"  [bold]{who}[/]  {escape(what)}")
console.print(
    "\n  02에서 여러분이 한 일과 같다: 신뢰구간으로 판정하고, 새로 틀린 질문을 의심하고,"
)
console.print(
    "  닿지 않는 범위(31위 밖)를 적고, 다음 가설을 골랐다. 달라진 것은 그 사이를 누가 채우느냐다"
)
console.print(f"\n원문: [dim]{escape(rel(RUN))}/02_plan.md · 04_results.md · 05_retro.md[/]")
