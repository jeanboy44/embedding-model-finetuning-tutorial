"""통합 테스트."""

import numpy as np
import pytest

from ragkit.retrieval import DocumentStore, retrieve


def test_document_store() -> None:
    """DocumentStore에 문서를 추가하고 상태를 확인한다."""
    store = DocumentStore()
    embeddings = np.random.randn(2, 384)
    store.add_documents(["doc1", "doc2"], embeddings)
    assert len(store.documents) == 2
    assert store.embeddings.shape == (2, 384)


def test_retrieve() -> None:
    """retrieve 함수가 올바른 수의 결과를 반환하는지 확인한다."""
    store = DocumentStore()
    embeddings = np.random.randn(3, 384)
    store.add_documents(["doc1", "doc2", "doc3"], embeddings)

    query_emb = np.random.randn(1, 384)
    results = retrieve(store, query_emb, k=2)
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)


def test_retrieve_uses_cosine_similarity() -> None:
    """정규화되지 않은 벡터에서도 코사인 유사도 순서와 값을 반환한다."""
    store = DocumentStore()
    embeddings = np.array([[10.0, 0.0], [1.0, 1.0], [0.0, 3.0]])
    store.add_documents(["x", "diag", "y"], embeddings)

    results = retrieve(store, np.array([2.0, 0.1]), k=3)

    assert [doc for doc, _ in results] == ["x", "diag", "y"]
    expected = 2.0 / np.linalg.norm([2.0, 0.1])
    assert results[0][1] == pytest.approx(expected)


def test_document_store_save_load(tmp_path) -> None:
    """DocumentStore의 save/load 라운드트립을 확인한다."""
    store = DocumentStore()
    embeddings = np.random.randn(2, 384)
    store.add_documents(["doc1", "doc2"], embeddings)

    store.save(tmp_path)

    store2 = DocumentStore()
    store2.load(tmp_path)
    assert len(store2.documents) == 2
    assert store2.embeddings.shape == (2, 384)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
