"""
실습 0-2 (3교시): 완성품 먼저 써 보기
=====================================

학습 목표:
- 강의 끝에 만들 결과물(법령 검색 + 웹 · CLI · MCP)을 먼저 써 본다
- 같은 질문을 학습 전 e5와 파인튜닝한 e5(r001_A)로 검색해 top-3를 나란히 본다
- 키워드 질문은 학습 전 모델도 잘 찾지만, 일상 말투의 상황 질문에서 차이가 커진다는 것을 확인한다
- 파인튜닝해도 못 찾는 질문이 있다는 것도 함께 본다 (실습 3의 오답 분석으로 이어짐)

검색은 앱(api · search-cli · mcp)과 같은 입구인 `ragkit.service.Searcher`를 쓴다.
서버는 띄우지 않는다. 마지막에 제품 세 가지를 띄우는 명령과 실습 5 스크립트를 안내한다.
받은 모델·인덱스로 검색만 하므로 `--run` 모드는 없다.

사전 준비 (uv run python lecture/03_setup/01_check_env.py 로 한 번에 확인):
    models/multilingual-e5-small, models/finetuned/r001_A        # 강사 Drive
    data/processed/index/multilingual-e5-small.sqlite            # 강사 Drive (또는 uv run ragkit index)
    data/processed/index/r001_A-<학습 시각>.sqlite                # 강사 Drive
    파인튜닝 모델은 ONNX로 바꾸지 않았으면 torch로 연다 (uv sync --all-packages --all-extras)

실행:
    uv run python lecture/03_setup/02_try_product.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "MLFLOW_DISABLE_AGENT_HINT", "1"
)  # mlflow가 설치돼 있으면 나오는 안내 끄기

from ragkit.config import get_settings
from ragkit.retrieval import default_index_path
from ragkit.service import Searcher

ROOT = Path(__file__).resolve().parents[2]
settings = get_settings()
FINETUNED = settings.models_dir / "finetuned" / "r001_A"
SPLIT_META = settings.data_dir / "splits" / "split_meta.json"
COMPARISON = (
    settings.experiments_dir
    / "exp_010_lecture_comparison"
    / "results"
    / "comparison.md"
)
K = 3

# (질문, 유형, 정답으로 볼 조 id). 유형은 스킬이 만든 질문의 세 가지 말투와 같다.
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
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def stop(message: str) -> None:
    print(f"\n{message}")
    print("준비 상태 점검: uv run python lecture/03_setup/01_check_env.py")
    sys.exit(1)


def open_searchers() -> tuple[Searcher, Searcher, str]:
    """학습 전 e5와 파인튜닝 모델을 연다. 파인튜닝 모델은 ONNX가 없으면 torch로 연다."""
    if not (FINETUNED / "config.json").exists():
        stop(f"파인튜닝 모델이 없습니다: {rel(FINETUNED)} (강사 Drive에서 받는다)")
    backend = "onnx" if (FINETUNED / "onnx" / "model.onnx").exists() else "torch"
    try:
        base = Searcher.open()
        tuned = Searcher.open(checkpoint=FINETUNED, backend=backend)
    except FileNotFoundError as e:
        stop(f"모델 또는 인덱스가 없습니다.\n{e}")
    except ImportError as e:
        stop(f"torch가 없어 파인튜닝 모델을 열 수 없습니다.\n{e}")
    return base, tuned, backend


def rank_of(searcher: Searcher, query: str, answer: str, depth: int = 100) -> str:
    hits = searcher.search(query, k=depth)
    for i, hit in enumerate(hits, 1):
        if (hit.metadata.get("parent_id") or hit.id) == answer:
            return f"{i}위"
    return f"{depth}위 밖"


def test_laws() -> set[str]:
    """학습 때 보지 않은 법령 (법령 단위 분할의 test)."""
    if not SPLIT_META.exists():
        return set()
    return set(json.loads(SPLIT_META.read_text(encoding="utf-8"))["test"]["laws"])


# ============================================================
# 1. 모델 두 개 열기
# ============================================================
section("1. 학습 전 e5 vs 파인튜닝 e5 (r001_A)")
start = time.perf_counter()
base, tuned, backend = open_searchers()
print(f"학습 전  {base.model_key:28s} 인덱스 {base.doc_count:,}개 조각 (onnx)")
print(f"파인튜닝 {tuned.model_key:28s} 인덱스 {tuned.doc_count:,}개 조각 ({backend})")
print(f"  인덱스 파일: {rel(default_index_path(tuned.model_key))} 처럼")
print("  모델마다 따로 있다 (같은 조문도 모델이 다르면 벡터가 다르다)")
print(f"모델·인덱스 열기 {time.perf_counter() - start:.1f}초")


# ============================================================
# 2. 같은 질문, 두 모델
# ============================================================
section(f"2. 같은 질문을 두 모델로 검색 (top-{K})")
print(
    "점수 크기는 모델마다 달라 서로 비교하지 않는다. 순위만 본다. ★ = 정답으로 볼 조문"
)
unseen = test_laws()
for query, qtype, answer in QUESTIONS:
    law = answer.split("_")[0]
    note = " · 학습 때 보지 않은 법령" if law in unseen else ""
    print(f'\n질문 [{qtype}] "{query}"')
    print(f"  정답으로 볼 조문: {answer}{note}")
    for label, searcher in (("학습 전", base), ("파인튜닝", tuned)):
        print(f"  [{label}] 정답 순위 {rank_of(searcher, query, answer)}")
        for hit in searcher.search(query, k=K):
            mark = "★" if (hit.metadata.get("parent_id") or hit.id) == answer else " "
            print(f"    {mark} {hit.score:.3f}  {hit.metadata['title']}")


# ============================================================
# 3. 전체 test 질문에서는? (실험 010 결과 파일)
# ============================================================
section("3. 질문 몇 개가 아니라 test 전체로 보면 (실험 010)")
if not COMPARISON.exists():
    print(f"결과 파일이 없습니다: {rel(COMPARISON)} (실습 3에서 다시 본다)")
else:
    # comparison.md의 표(모델별 지표, 질문 유형별 R@5)에서 학습 전 e5와 r001_A 행만 읽는다
    rows: dict[str, dict[str, str]] = {}
    header: list[str] = []
    for line in COMPARISON.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        if cells[0] == "모델":
            header = cells
        else:
            rows.setdefault(cells[0], {}).update(zip(header, cells))
    cols = ["R@5", "keyword", "question", "situation"]
    print(
        f"  test R@5 (전체 · 질문 유형별, n=질문 수)\n  {'모델':32s}"
        + "".join(f"{c:>16s}" for c in cols)
    )
    for name in (settings.embedding_model_name, FINETUNED.name):
        values = rows.get(name, {})
        print(f"  {name:34s}" + "".join(f"{values.get(c, '-'):>16s}" for c in cols))
    print(f"  출처: {rel(COMPARISON)}")
print("  → 키워드 질문보다 상황 설명(situation) 질문에서 학습 전 모델이 더 약하다")


# ============================================================
# 4. 제품 세 가지 (실습 5에서 직접 띄운다)
# ============================================================
section("4. 완성품 세 가지: 웹 · CLI · MCP")
ckpt = rel(FINETUNED)
print(f"""모두 같은 Searcher를 쓰고, --checkpoint {ckpt} --backend torch 를 붙이면
파인튜닝 모델로 바뀐다. 이 스크립트는 서버를 띄우지 않는다.

[CLI] 터미널에서 바로 검색
  uv run --package ragkit-search ragkit-search search "야간 근로 수당" -k 3
  uv run --package ragkit-search ragkit-search search "야간 근로 수당" -k 3 \\
      --checkpoint {ckpt} --backend torch
  → 실습 5: uv run python lecture/08_product/01_search_cli.py

[MCP] Claude Code 같은 에이전트가 법령을 검색하는 도구로 등록
  claude mcp add ragkit-law -e RAGKIT_PROJECT_ROOT={ROOT} -- \\
      uv run --directory {ROOT} --package ragkit-mcp ragkit-mcp
  → 실습 5: uv run python lecture/08_product/03_mcp_server.py

[웹] 법령 노트 화면 (API 서버가 빌드한 화면을 함께 제공)
  cd apps/web && pnpm install && pnpm build && cd ../..
  uv run --package ragkit-api ragkit-api --web-dist apps/web/dist   # http://127.0.0.1:8000
  → 실습 5: uv run python lecture/08_product/04_web_app.py

답변 생성(ask, 웹 채팅)은 .env의 GEMINI_API_KEY가 있어야 한다. 없으면 검색 결과만 보여 준다.""")

print("""
정리:
- 같은 코퍼스, 같은 크기(e5-small)의 모델인데 학습한 모델이 일상 말투 질문에서 정답을 더 위로 올린다
- 그래도 못 찾는 질문이 남는다 → 실습 2(평가)와 실습 3(오답 분석)에서 숫자로 본다
- 다음: uv run python lecture/03_setup/03_why_rag.py (왜 검색이 필요한가)
""")
