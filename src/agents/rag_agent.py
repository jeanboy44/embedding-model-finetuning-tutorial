"""RAG 에이전트 — 검색 + 생성 파이프라인."""

from collections.abc import Callable

import numpy as np

from src.retrieval import DocumentStore, retrieve


def run_rag(
    query: str,
    store: DocumentStore,
    embed_fn: Callable[[list[str]], np.ndarray],
    generate_fn: Callable[[str], str],
    *,
    k: int = 3,
) -> dict[str, str | list[str]]:
    """RAG 파이프라인을 실행한다: 검색 → 생성.

    Args:
        query: 사용자 질문.
        store: 문서 저장소.
        embed_fn: 텍스트 → 임베딩 함수 (의존성 주입).
        generate_fn: 프롬프트 → 텍스트 생성 함수 (의존성 주입).
        k: 검색할 문서 수.

    Returns:
        query, retrieved_documents, answer를 포함하는 딕셔너리.

    Example:
        >>> from src.embeddings import create_embedding_fn
        >>> from src.models import generate_text
        >>> embed = create_embedding_fn("thenlper/gte-small")
        >>> result = run_rag("질문", store, embed, generate_text)
    """
    query_embedding = embed_fn([query])

    retrieved_docs = retrieve(store, query_embedding, k=k)
    doc_texts = [doc for doc, _ in retrieved_docs]

    context = "\n".join([f"- {doc}" for doc in doc_texts])
    prompt = (
        f"Based on the following context, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    answer = generate_fn(prompt)

    return {
        "query": query,
        "retrieved_documents": doc_texts,
        "answer": answer,
    }
