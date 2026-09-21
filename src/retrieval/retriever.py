"""문서 검색기."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.document_store import DocumentStore


def retrieve(
    store: DocumentStore,
    query_embeddings: np.ndarray,
    k: int = 5,
) -> list[tuple[str, float]]:
    """저장소에서 상위 k개의 유사 문서를 검색한다.

    Args:
        store: 검색 대상 문서 저장소.
        query_embeddings: (1, dim) 또는 (dim,) 형태의 쿼리 임베딩.
        k: 반환할 최대 문서 수.

    Returns:
        (문서 텍스트, 유사도 점수) 튜플의 리스트. 유사도 내림차순 정렬.
    """
    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.reshape(1, -1)

    similarities = cosine_similarity(query_embeddings, store.embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]

    return [(store.documents[idx], float(similarities[idx])) for idx in top_indices]
