"""문서 검색기."""

import numpy as np

from ragkit.retrieval.document_store import DocumentStore


def _normalize(x: np.ndarray) -> np.ndarray:
    """행 단위 L2 정규화. 영벡터는 0으로 남긴다."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


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

    # 코사인 유사도 = 정규화한 벡터의 내적 (scikit-learn 없이 numpy로 계산)
    similarities = (_normalize(query_embeddings) @ _normalize(store.embeddings).T)[0]
    top_indices = np.argsort(similarities)[::-1][:k]

    return [(store.documents[idx], float(similarities[idx])) for idx in top_indices]
