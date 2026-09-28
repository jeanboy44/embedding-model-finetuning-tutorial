"""MCP Server skeleton — 학생 확장용 구조 제공."""

from collections.abc import Callable

import numpy as np

from ragkit.rag import run_rag
from ragkit.embeddings import format_queries
from ragkit.retrieval import DocumentStore, retrieve


def embed_tool(
    text: str,
    embed_fn: Callable[[list[str]], np.ndarray],
) -> dict:
    """Tool: 임베딩을 생성한다.

    Args:
        text: 임베딩할 텍스트.
        embed_fn: 텍스트 → 임베딩 함수.

    Returns:
        shape, first_5를 포함하는 딕셔너리.
    """
    embedding = embed_fn([text])
    return {
        "shape": str(embedding.shape),
        "first_5": embedding[0][:5].tolist() if len(embedding) > 0 else [],
    }


def retrieve_tool(
    query: str,
    store: DocumentStore,
    embed_fn: Callable[[list[str]], np.ndarray],
    k: int = 5,
) -> dict:
    """Tool: 문서를 검색한다.

    Args:
        query: 검색 쿼리.
        store: 문서 저장소.
        embed_fn: 텍스트 → 임베딩 함수.
        k: 반환할 문서 수.

    Returns:
        query, results를 포함하는 딕셔너리.
    """
    query_embedding = embed_fn(format_queries([query]))
    results = retrieve(store, query_embedding, k=k)
    return {
        "query": query,
        "results": [{"doc": doc, "score": float(score)} for doc, score in results],
    }


def rag_tool(
    query: str,
    store: DocumentStore,
    embed_fn: Callable[[list[str]], np.ndarray],
    generate_fn: Callable[[str], str],
) -> dict:
    """Tool: RAG (검색 + 생성).

    Args:
        query: 사용자 질문.
        store: 문서 저장소.
        embed_fn: 텍스트 → 임베딩 함수.
        generate_fn: 프롬프트 → 텍스트 생성 함수.

    Returns:
        query, retrieved_documents, answer를 포함하는 딕셔너리.
    """
    return run_rag(query, store, embed_fn, generate_fn)
