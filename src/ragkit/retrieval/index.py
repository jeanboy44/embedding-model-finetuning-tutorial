"""코퍼스 인덱스 생성·캐시와 id 포함 검색.

전체 코퍼스(약 2.6만 청크) 임베딩은 수 분이 걸리므로 한 번 만든 인덱스를 캐시해
비교 실습·평가·앱이 같이 쓴다.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragkit.embeddings.prefix import format_passages
from ragkit.retrieval.document_store import DocumentStore
from ragkit.retrieval.retriever import _normalize

EmbedFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class SearchHit:
    """검색 결과 한 건."""

    id: str
    text: str
    score: float


def _corpus_hash(ids: list[str], texts: list[str]) -> str:
    h = hashlib.sha256()
    for doc_id, text in zip(ids, texts, strict=True):
        h.update(doc_id.encode())
        h.update(b"\0")
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()[:12]


def build_index(
    ids: list[str],
    texts: list[str],
    embed_fn: EmbedFn,
    *,
    cache_dir: Path | None = None,
    cache_key: str = "",
    batch_size: int = 64,
) -> DocumentStore:
    """문서를 임베딩해 인덱스(DocumentStore)를 만든다. 캐시가 있으면 읽기만 한다.

    Args:
        ids: 문서 id 리스트.
        texts: 문서 텍스트 리스트 (앞 문구 없이). passage 앞 문구는 여기서 붙인다.
        embed_fn: 임베딩 함수 (ragkit.embeddings.create_embedding_fn의 반환값).
        cache_dir: 캐시 폴더. None이면 캐시하지 않는다.
        cache_key: 모델을 구분하는 이름 (예: "multilingual-e5-small-onnx").
            모델이 바뀌면 다른 값을 줘야 캐시가 섞이지 않는다.
        batch_size: 임베딩 배치 크기.

    Returns:
        metadata에 {"id": ...}가 들어간 DocumentStore.
    """
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{cache_key}_{_corpus_hash(ids, texts)}"
        if (path / "embeddings.npy").exists():
            store = DocumentStore()
            store.load(path)
            return store

    embeddings = embed_fn(format_passages(texts), batch_size=batch_size)
    store = DocumentStore()
    store.add_documents(list(texts), embeddings, [{"id": i} for i in ids])
    if path is not None:
        store.save(path)
    return store


def search(store: DocumentStore, query_embedding: np.ndarray, k: int = 5) -> list[SearchHit]:
    """코사인 유사도 상위 k개 문서를 id와 함께 반환한다.

    Args:
        store: build_index로 만든 인덱스.
        query_embedding: (dim,) 또는 (1, dim) 쿼리 임베딩 (query 앞 문구를 붙여 임베딩한 것).
        k: 반환할 문서 수.

    Returns:
        점수 내림차순 SearchHit 리스트.
    """
    q = np.asarray(query_embedding).reshape(1, -1)
    scores = (_normalize(q) @ _normalize(store.embeddings).T)[0]
    top = np.argsort(scores)[::-1][:k]
    return [
        SearchHit(id=store.metadata[i]["id"], text=store.documents[i], score=float(scores[i]))
        for i in top
    ]
