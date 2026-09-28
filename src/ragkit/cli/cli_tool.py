"""ragkit CLI 도구 — cyclopts 기반."""

from pathlib import Path

import cyclopts

from ragkit.rag import run_rag
from ragkit.config import get_settings
from ragkit.embeddings import create_embedding_fn, format_queries
from ragkit.models import generate_text
from ragkit.retrieval import DocumentStore, retrieve

app = cyclopts.App(name="ragkit", help="ragkit: 임베딩 검색 기반 RAG CLI")


@app.command
def embed(
    text: list[str],
    model: str | None = None,
) -> None:
    """텍스트의 임베딩을 생성한다.

    Args:
        text: 임베딩할 텍스트 (여러 개 가능).
        model: 임베딩 모델 이름. None이면 Settings 기본값 사용.
    """
    if not text:
        print("Error: --text를 최소 하나 입력하세요.")
        return

    settings = get_settings()
    embed_fn = create_embedding_fn(model or settings.embedding_model_name)
    embeddings = embed_fn(list(text))

    for t, emb in zip(text, embeddings):
        print(f"Text: {t}")
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding (first 5 dims): {emb[:5]}")
        print()


@app.command
def search(
    query: str,
    model: str | None = None,
) -> None:
    """쿼리 기반으로 문서를 검색한다.

    Args:
        query: 검색 쿼리.
        model: 임베딩 모델 이름. None이면 Settings 기본값 사용.
    """
    store = _load_document_store()
    if store is None:
        return

    settings = get_settings()
    embed_fn = create_embedding_fn(model or settings.embedding_model_name)
    query_embedding = embed_fn(format_queries([query]))
    results = retrieve(store, query_embedding, k=5)

    print(f"\nQuery: {query}\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{score:.4f}] {doc}")


@app.command
def rag(
    query: str,
    model: str | None = None,
) -> None:
    """RAG: 문서 검색 후 답변을 생성한다.

    Args:
        query: 사용자 질문.
        model: 임베딩 모델 이름. None이면 Settings 기본값 사용.
    """
    store = _load_document_store()
    if store is None:
        return

    settings = get_settings()
    embed_fn = create_embedding_fn(model or settings.embedding_model_name)
    result = run_rag(query, store, embed_fn, generate_text)

    print(f"\nQuestion: {result['query']}\n")
    print("Retrieved Documents:")
    for i, doc in enumerate(result["retrieved_documents"], 1):
        print(f"  {i}. {doc}")
    print(f"\nAnswer:\n{result['answer']}")


def _load_document_store() -> DocumentStore | None:
    """문서 저장소를 로드한다.

    Returns:
        로드된 DocumentStore. 실패 시 None.
    """
    store_path = Path("experiments/results")
    if not store_path.exists():
        print("Error: Document store not found")
        return None

    store = DocumentStore()
    try:
        store.load(store_path)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Error: Could not load document store: {e}")
        return None
    return store


if __name__ == "__main__":
    app()
