"""통합 테스트."""

import numpy as np
import pytest

from src.retrieval import DocumentStore, retrieve


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
