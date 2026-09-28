"""ragkit-search 명령: 법령 검색·답변·법령 목록·조문 보기 (표준 라이브러리 출력만)."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cyclopts
from cyclopts import Parameter

from ragkit.retrieval import SearchHit
from ragkit.service import DeltaEvent, DoneEvent, HitsEvent, Searcher

app = cyclopts.App(
    name="ragkit-search",
    help="법령 검색 CLI: 질문과 가까운 조문을 찾고, 근거와 함께 답한다.",
    default_parameter=Parameter(negative=()),  # --no-json·--empty-law 같은 반대 플래그는 만들지 않는다
)

SNIPPET_CHARS = 120


@Parameter(name="*", group="모델·인덱스")
@dataclass
class SearcherOptions:
    """모델·인덱스 선택 (주지 않으면 Settings / .env 기본값).

    Args:
        model: 임베딩 모델 이름. 기본값 Settings.embedding_model_name.
        checkpoint: 파인튜닝한 모델 폴더.
        backend: onnx | torch | st. 기본값 Settings.embedding_backend.
        index: 인덱스 파일. 기본값 data/processed/index/<모델 키>.sqlite.
    """

    model: str | None = None
    checkpoint: Path | None = None
    backend: str | None = None
    index: Path | None = None


def _open_searcher(options: SearcherOptions) -> Searcher:
    """Searcher를 연다. 테스트는 이 함수를 바꿔 끼운다."""
    return Searcher.open(
        model=options.model, checkpoint=options.checkpoint, backend=options.backend, index_path=options.index
    )


def _fail(message: str) -> int:
    print(f"오류: {message}", file=sys.stderr)
    return 1


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 1] + "…"


def _title(hit: SearchHit) -> str:
    return hit.metadata.get("title") or hit.id


def _print_json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _print_sources(hits: list[SearchHit]) -> None:
    print("\n근거 조문:")
    for n, hit in enumerate(hits, 1):
        print(f"  [{n}] {_title(hit)} ({hit.id})")


@app.command
def search(
    query: str,
    /,
    *,
    k: Annotated[int, Parameter(alias="-k")] = 5,
    law: list[str] | None = None,
    json_: Annotated[bool, Parameter(name="--json")] = False,
    options: SearcherOptions | None = None,
) -> int:
    """질문과 가까운 조문을 찾는다.

    Args:
        query: 검색할 질문. 예: "야간 근로 수당".
        k: 보여 줄 조문 수.
        law: 이 법령 안에서만 찾는다 (여러 번 줄 수 있다). 정확한 이름은 `laws` 명령으로 확인.
        json_: 결과를 JSON으로 출력한다.
    """
    try:
        hits = _open_searcher(options or SearcherOptions()).search(query, k=k, laws=law)
    except (FileNotFoundError, ValueError) as e:
        return _fail(str(e))
    if json_:
        _print_json([hit.to_dict() for hit in hits])
        return 0
    if not hits:
        print("검색 결과가 없습니다.")
    for rank, hit in enumerate(hits, 1):
        print(f"{rank:>2}. [{hit.score:.3f}] {_title(hit)} ({hit.id})")
        print(f"    {_snippet(hit.text)}")
    return 0


@app.command
def ask(
    question: str,
    /,
    *,
    k: Annotated[int, Parameter(alias="-k")] = 5,
    law: list[str] | None = None,
    options: SearcherOptions | None = None,
) -> int:
    """찾은 조문만 근거로 답한다 (GEMINI_API_KEY 필요). 답을 받는 대로 출력하고 근거 조문을 덧붙인다.

    Args:
        question: 질문.
        k: 근거로 쓸 조문 수.
        law: 이 법령 안에서만 찾는다 (여러 번 줄 수 있다).
    """
    hits: list[SearchHit] = []
    error: str | None = None
    try:
        for event in _open_searcher(options or SearcherOptions()).answer_stream(question, k=k, laws=law):
            if isinstance(event, HitsEvent):
                hits = event.hits
            elif isinstance(event, DeltaEvent):
                print(event.text, end="", flush=True)
            elif isinstance(event, DoneEvent):
                error = event.error
                if event.answer:
                    print()
    except (FileNotFoundError, ValueError) as e:
        return _fail(str(e))
    if hits:
        _print_sources(hits)
    return _fail(error) if error else 0


@app.command
def laws(
    *,
    theme: str | None = None,
    json_: Annotated[bool, Parameter(name="--json")] = False,
    options: SearcherOptions | None = None,
) -> int:
    """인덱스에 든 법령 목록 (`--law`에 쓸 정확한 이름).

    Args:
        theme: 이 테마의 법령만 (예: youth).
        json_: 결과를 JSON으로 출력한다.
    """
    try:
        infos = _open_searcher(options or SearcherOptions()).list_laws(theme)
    except (FileNotFoundError, ValueError) as e:
        return _fail(str(e))
    if json_:
        _print_json([vars(info) for info in infos])
        return 0
    if not infos:
        print("법령이 없습니다.")
    for info in infos:
        print(f"{info.law_name}  ({info.law_type} · {info.theme} · 조각 {info.doc_count}개)")
    return 0


@app.command
def show(
    doc_id: str,
    /,
    *,
    article: bool = False,
    options: SearcherOptions | None = None,
) -> int:
    """조문 조각 하나를 전문으로 본다.

    Args:
        doc_id: 조문 id (search 결과의 괄호 안 값).
        article: 같은 조의 조각(항·호)을 모두 보여 준다.
    """
    try:
        searcher = _open_searcher(options or SearcherOptions())
    except (FileNotFoundError, ValueError) as e:
        return _fail(str(e))
    hit = searcher.get(doc_id)
    if hit is None:
        return _fail(f"없는 조문 id: {doc_id}")
    pieces = searcher.get_article(hit.metadata.get("parent_id") or hit.id) if article else [hit]
    for i, piece in enumerate(pieces):
        if i:
            print()
        print(f"{_title(piece)} ({piece.id})")
        print(piece.text)
    if url := hit.metadata.get("source_url"):
        print(f"\n출처: {url}")
    return 0
