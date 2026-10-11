"""
실습 0-2 (3교시): 완성품 먼저 써 보기
=====================================

학습 목표:
- 강의 끝에 만들 결과물(법령 검색 + 웹 · CLI · MCP)을 먼저 써 본다
- 같은 질문을 학습 전 e5와 파인튜닝한 e5(002_hard_negatives)로 검색해 top-3를 나란히 본다
- 키워드 질문은 학습 전 모델도 잘 찾지만, 일상 말투의 상황 질문에서 차이가 커진다는 것을 확인한다
- 파인튜닝해도 못 찾는 질문이 있다는 것도 함께 본다 (6교시 실습 2의 오답 분석으로 이어짐)

검색은 앱(api · search-cli · mcp)과 같은 입구인 `ragkit.service.Searcher`를 쓴다.
서버는 띄우지 않는다. 마지막에 제품 세 가지를 띄우는 명령과 실습 4(8교시) 스크립트를 안내한다.
받은 모델·인덱스로 검색만 하므로 `--run` 모드는 없다.

무엇을 하나 (강사 PC 약 20초):
    1. Searcher.open으로 학습 전 e5(onnx)와 002_hard_negatives(torch)를 열고, 인덱스 조각 수와 여는 데 걸린 시간을 찍는다
    2. 질문 넷을 두 모델로 Searcher.search 해 정답 조문의 순위(100위까지)와 top-3를 나란히 찍는다. ★가 정답 조문
    3. 실험 002를 끝내며 test로 한 번 잰 결과(조 단위 1,863문항)에서 두 모델의 R@5(전체 · 질문 유형별)를 찍는다
    4. 완성품 세 가지(CLI · MCP · 웹)를 띄우는 명령과 8교시 스크립트를 안내한다. 서버는 띄우지 않는다

사전 준비 (uv run python lecture/03_setup/01_doctor.py 로 한 번에 확인):
    models/multilingual-e5-small                                 # scripts/download_model_hf.py
    models/finetuned/002_hard_negatives + 그 인덱스                           # scripts/finetuned_drive.py download
    data/processed/index/multilingual-e5-small.sqlite            # (위 명령이 함께 받는다. 또는 uv run ragkit index)
    data/processed/index/002_hard_negatives-<학습 시각>.sqlite                # (위 명령이 함께 받는다)
    파인튜닝 모델은 ONNX로 바꾸지 않았으면 torch로 연다 (uv sync --all-packages --all-extras)

실행:
    uv run python lecture/03_setup/02_try_product.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "MLFLOW_DISABLE_AGENT_HINT", "1"
)  # mlflow가 설치돼 있으면 나오는 안내 끄기

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ragkit.config import get_settings
from ragkit.retrieval import default_index_path, model_key
from ragkit.service import Searcher

ROOT = Path(__file__).resolve().parents[2]
console = Console(emoji=False)
settings = get_settings()
FINETUNED = settings.models_dir / "finetuned" / "002_hard_negatives"
# 실험 002를 끝내며 test(조 단위 분할, 1,863문항)로 한 번 잰 doc R@5. 결과 JSON은 나눠 주지 않아 숫자를 옮겨 둔다.
# (모델, 전체, keyword, question, situation). 질문 수는 keyword 399 · question 701 · situation 763
TEST_R5 = [
    ("학습 전 e5", 0.545, 0.739, 0.573, 0.417),
    ("002_hard_negatives", 0.773, 0.875, 0.795, 0.701),
]
K = 3  # 화면에 보여 줄 상위 조문 수. 정답 순위는 K와 상관없이 100위까지 찾는다

# (질문, 유형, 정답으로 볼 조 id). 유형은 스킬이 만든 질문의 세 가지 말투와 같다.
# keyword(법률 용어) · question(궁금한 점) · situation(상황 설명). 뒤로 갈수록 학습 전 모델이 약하다.
QUESTIONS = [
    ("야간 근로 수당", "keyword", "근로기준법_법률_제56조"),
    (
        "편의점 알바 3개월 했는데 주휴수당 받을 수 있나요?",
        "situation",
        "근로기준법_법률_제55조",
    ),
    (
        "회사 그만뒀는데 퇴직금은 언제까지 받아야 해요?",
        "question",
        "근로자퇴직급여보장법_법률_제9조",
    ),
    (
        "방문판매로 산 정수기 환불하고 싶어요",
        "situation",
        "방문판매등에관한법률_법률_제8조",
    ),
]


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold]{escape(title)}[/]", style="blue")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def stop(message: str) -> None:
    console.print()
    console.print(
        Panel(
            f"{escape(message)}\n준비 상태 점검: [dim]uv run python lecture/03_setup/01_doctor.py[/]",
            title="필요한 산출물이 없습니다",
            border_style="red",
        )
    )
    sys.exit(1)


def open_searchers() -> tuple[Searcher, Searcher, str]:
    """학습 전 e5와 파인튜닝 모델을 연다. 파인튜닝 모델은 ONNX가 없으면 torch로 연다."""
    if not (FINETUNED / "config.json").exists():
        stop(
            f"파인튜닝 모델이 없습니다: {rel(FINETUNED)} (uv run python scripts/finetuned_drive.py download)"
        )
    # 받은 002에는 onnx/ 폴더가 없어 torch로 열린다 (7교시에서 ONNX · INT8로 바꾼다)
    backend = "onnx" if (FINETUNED / "onnx" / "model.onnx").exists() else "torch"
    try:
        base = Searcher.open()  # 설정 기본값: multilingual-e5-small + ONNX + 그 인덱스
        # 파인튜닝 모델. 인덱스는 모델 키(폴더 이름 + 가중치 수정 시각)로 자동 선택
        tuned = Searcher.open(checkpoint=FINETUNED, backend=backend)
    except FileNotFoundError as e:
        stop(f"모델 또는 인덱스가 없습니다.\n{e}")
    except ImportError as e:
        stop(f"torch가 없어 파인튜닝 모델을 열 수 없습니다.\n{e}")
    return base, tuned, backend


def rank_of(searcher: Searcher, query: str, answer: str, depth: int = 100) -> str:
    """정답 조문이 몇 위에 처음 나오는지. 조 단위라 그 조의 어느 항 · 호가 떠도 정답으로 센다."""
    hits = searcher.search(query, k=depth)
    for i, hit in enumerate(hits, 1):
        # 항 · 호 조각은 parent_id가 조 id, 조 전체가 한 조각이면 id가 곧 조 id
        if (hit.metadata.get("parent_id") or hit.id) == answer:
            return f"{i}위"
    return f"{depth}위 밖"


# ============================================================
# 1. 모델 두 개 열기
# ============================================================
# 앱과 같은 입구 Searcher.open으로 두 모델을 연다. 인덱스는 모델마다 따로 있고 조각 수는 같다(25,967).
# 출력에서 볼 것: 두 줄의 인덱스 조각 수가 같은지, 여는 데 걸린 시간(강사 PC 약 11초).
# 파인튜닝 줄에는 폴더 이름을 찍는다. 인덱스 안에 적힌 모델 키는 학습 당시 run 이름(실험 번호를 바꾸기 전 이름)이라 헷갈린다.
section("1. 학습 전 e5 vs 파인튜닝 e5 (002_hard_negatives)")
start = time.perf_counter()
base, tuned, backend = open_searchers()
console.print(
    f"학습 전  {escape(f'{base.model_key:28s}')} 인덱스 [bold]{base.doc_count:,}개[/] 조각 (onnx)"
)
console.print(
    f"파인튜닝 {escape(f'{FINETUNED.name:28s}')} 인덱스 [bold]{tuned.doc_count:,}개[/] 조각 ({backend})"
)
console.print(
    f"  [dim]인덱스 파일: {escape(rel(default_index_path(model_key(str(FINETUNED), FINETUNED))))} 처럼[/]",
    soft_wrap=True,
)
console.print("  [dim]모델마다 따로 있다 (같은 조문도 모델이 다르면 벡터가 다르다)[/]")
console.print(f"모델·인덱스 열기 [bold]{time.perf_counter() - start:.1f}초[/]")


# ============================================================
# 2. 같은 질문, 두 모델
# ============================================================
# 질문마다 두 모델로 Searcher.search를 두 번 부른다: 100위까지 찾아 정답 순위, 다시 top-3를 화면에.
# 출력에서 볼 것: 키워드 질문은 둘 다 상위, 상황 질문은 학습 전 65위 → 파인튜닝 2위처럼 차이가 크다.
# 방문판매법 질문은 파인튜닝해도 16위라 못 찾는다 (6교시 오답 분석으로 이어진다).
section(f"2. 같은 질문을 두 모델로 검색 (top-{K})")
console.print(
    "[dim]점수 크기는 모델마다 달라 서로 비교하지 않는다. 순위만 본다.[/] [green]★[/] = 정답으로 볼 조문"
)
for query, qtype, answer in QUESTIONS:
    console.print(f'\n[bold]질문[/] {escape(f"[{qtype}]")} "{escape(query)}"')
    console.print(f"  정답으로 볼 조문: {escape(answer)}")
    for label, searcher in (("학습 전", base), ("파인튜닝", tuned)):
        console.print(
            f"  {escape(f'[{label}]')} 정답 순위 [bold]{rank_of(searcher, query, answer)}[/]"
        )
        for hit in searcher.search(query, k=K):
            is_answer = (hit.metadata.get("parent_id") or hit.id) == answer
            mark = "[green]★[/]" if is_answer else " "
            console.print(
                f"    {mark} {hit.score:.3f}  {escape(hit.metadata['title'])}",
                highlight=False,
            )


# ============================================================
# 3. 전체 test 질문에서는? (실험 002의 test 한 번)
# ============================================================
# 검색은 하지 않고, 강사가 실험을 끝내며 test로 한 번 잰 숫자(TEST_R5)를 찍는다.
# 조 단위 분할의 test 1,863문항: 54개 법령은 모두 학습에 있고, 학습 질문이 없던 조만 모았다.
# 전체 평가는 5 · 6교시에서 ragkit evaluate로 직접 한다.
# 출력에서 볼 것: R@5 0.545 → 0.773, situation 0.417 → 0.701로 상황 질문의 차이가 가장 크다.
section("3. 질문 몇 개가 아니라 test 전체로 보면 (실험 002)")
console.print("  test R@5 (조 단위 분할 1,863문항 · 질문 유형별)")
table = Table(box=box.SIMPLE_HEAD, show_edge=False)
table.add_column("모델")
for col in ("R@5", "keyword (399)", "question (701)", "situation (763)"):
    table.add_column(col, justify="right")
for name, *values in TEST_R5:
    table.add_row(escape(name), *(f"{v:.3f}" for v in values))
console.print(table)
console.print("  [dim]출처: 실험 002를 끝내며 test로 한 번 잰 결과 (002는 dev로만 골랐다)[/]")
console.print("  → 키워드 질문보다 상황 설명(situation) 질문에서 학습 전 모델이 더 약하고, 학습으로 가장 많이 오른다")


# ============================================================
# 4. 제품 세 가지 (8교시 실습 4에서 직접 띄운다)
# ============================================================
# 실행하지 않고 명령만 안내한다. 세 제품 모두 위와 같은 Searcher를 쓰므로
# --checkpoint 와 --backend torch 를 붙이면 파인튜닝 모델로 바뀐다는 점을 본다.
section("4. 완성품 세 가지: 웹 · CLI · MCP")
ckpt = rel(FINETUNED)
console.print(
    f"""모두 같은 Searcher를 쓰고, --checkpoint {ckpt} --backend torch 를 붙이면
파인튜닝 모델로 바뀐다. 이 스크립트는 서버를 띄우지 않는다.

[CLI] 터미널에서 바로 검색
  uv run --package ragkit-search ragkit-search search "야간 근로 수당" -k 3
  uv run --package ragkit-search ragkit-search search "야간 근로 수당" -k 3 --checkpoint {ckpt} --backend torch
  → 실습 4: uv run python lecture/08_product/01_search_cli.py

[MCP] Claude Code 같은 에이전트가 법령을 검색하는 도구로 등록
  claude mcp add ragkit-law -e RAGKIT_PROJECT_ROOT={ROOT} -- uv run --directory {ROOT} --package ragkit-mcp ragkit-mcp
  → 실습 4: uv run python lecture/08_product/03_mcp_server.py

[웹] 검색 화면 (API 서버가 HTML 한 장을 함께 제공, 빌드 없음)
  uv run --package ragkit-api ragkit-api --web-dist apps/web-lite   # http://127.0.0.1:8000
  → 실습 4: uv run python lecture/08_product/04_web_app.py

답변 생성(LLM)은 이 강의에서 쓰지 않는다. 세 제품 모두 검색 결과(근거 조문)까지 본다.""",
    markup=False,
    highlight=False,
    soft_wrap=True,
)

console.print(
    """
정리:
- 같은 코퍼스, 같은 크기(e5-small)의 모델인데 학습한 모델이 일상 말투 질문에서 정답을 더 위로 올린다
- 그래도 못 찾는 질문이 남는다 → 실습 1(평가)과 실습 2(오답 분석)에서 숫자로 본다
- 다음: lecture/03_setup/README.md의 명령으로 CLI에서 직접 학습 전후를 비교한다
""",
    highlight=False,
)
