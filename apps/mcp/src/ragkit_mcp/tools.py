"""MCP 도구 로직: Searcher를 첫 인자로 받는 순수 함수 (server.py는 등록만 한다).

반환값은 JSON으로 보내기 좋은 dict / list[dict]다.
"""

from ragkit.retrieval import SearchHit
from ragkit.service import Searcher


def _piece(hit: SearchHit) -> dict:
    return {"id": hit.id, "title": hit.metadata.get("title", ""), "text": hit.text}


def search_laws(searcher: Searcher, query: str, k: int = 5, laws: list[str] | None = None) -> list[dict]:
    """질문과 가까운 조문 조각 상위 k개.

    Args:
        searcher: 열린 Searcher.
        query: 검색할 질문 (자연어).
        k: 돌려줄 조각 수.
        laws: 이 법령들 안에서만 찾는다. 정확한 이름이어야 한다 (list_laws로 확인).

    Raises:
        ValueError: 모르는 법령 이름이 있을 때 (비슷한 이름을 안내한다).
    """
    return [
        {
            "id": hit.id,
            "title": hit.metadata.get("title", ""),
            "law_name": hit.metadata.get("law_name", ""),
            "article_no": hit.metadata.get("article_no", ""),
            "score": round(hit.score, 4),
            "text": hit.text,
            "source_url": hit.metadata.get("source_url", ""),
        }
        for hit in searcher.search(query, k=k, laws=laws)
    ]


def get_article(searcher: Searcher, doc_id: str) -> dict:
    """조각 id가 속한 조(條) 전체. 항·호로 나뉜 조문도 모든 조각을 순서대로 돌려준다.

    Args:
        searcher: 열린 Searcher.
        doc_id: search_laws 결과의 id.

    Raises:
        ValueError: 없는 id일 때.
    """
    hit = searcher.get(doc_id)
    if hit is None:
        raise ValueError(f"없는 조문 id: {doc_id} (search_laws 결과의 id를 쓰세요)")
    pieces = searcher.get_article(hit.metadata.get("parent_id") or hit.id) or [hit]
    return {
        "title": pieces[0].metadata.get("title", ""),
        "law_name": hit.metadata.get("law_name", ""),
        "source_url": hit.metadata.get("source_url", ""),
        "pieces": [_piece(p) for p in pieces],
    }


def list_laws(searcher: Searcher, theme: str | None = None) -> list[dict]:
    """인덱스에 든 법령 목록 (law_name, law_type, theme, doc_count).

    Args:
        searcher: 열린 Searcher.
        theme: 이 테마의 법령만 (예: youth). None이면 전체.
    """
    return [
        {"law_name": law.law_name, "law_type": law.law_type, "theme": law.theme, "doc_count": law.doc_count}
        for law in searcher.list_laws(theme)
    ]


def ask(searcher: Searcher, question: str, laws: list[str] | None = None, k: int = 5) -> dict:
    """검색한 조문만 근거로 한 Gemini 답변. 답의 [n]은 sources의 n번째 조문이다.

    Args:
        searcher: 열린 Searcher.
        question: 질문.
        laws: 이 법령들 안에서만 찾는다.
        k: 근거로 쓸 조문 수.

    Returns:
        {answer, sources: [{n, id, title}], error}. LLM을 못 쓰면 answer는 비고 error에 이유가 담긴다.
    """
    result = searcher.answer(question, k=k, laws=laws)
    return {
        "answer": result.answer,
        "sources": [
            {"n": n, "id": hit.id, "title": hit.metadata.get("title", "")} for n, hit in enumerate(result.hits, 1)
        ],
        "error": result.error,
    }
