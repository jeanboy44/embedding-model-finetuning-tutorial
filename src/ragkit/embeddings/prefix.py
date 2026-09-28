"""쿼리/문서 앞 문구 (e5 계열: "query: " / "passage: ")."""

from ragkit.config import get_settings


def format_queries(texts: list[str]) -> list[str]:
    """검색 쿼리에 모델이 요구하는 앞 문구를 붙인다 (e5: "query: ").

    Args:
        texts: 쿼리 텍스트 리스트.

    Returns:
        앞 문구가 붙은 텍스트 리스트.
    """
    prefix = get_settings().query_prefix
    return [f"{prefix}{t}" for t in texts]


def format_passages(texts: list[str]) -> list[str]:
    """검색 대상 문서에 모델이 요구하는 앞 문구를 붙인다 (e5: "passage: ").

    Args:
        texts: 문서 텍스트 리스트.

    Returns:
        앞 문구가 붙은 텍스트 리스트.
    """
    prefix = get_settings().passage_prefix
    return [f"{prefix}{t}" for t in texts]
