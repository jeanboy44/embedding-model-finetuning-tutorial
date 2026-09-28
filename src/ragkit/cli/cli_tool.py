"""ragkit CLI 도구 — cyclopts 기반."""

from pathlib import Path

import cyclopts

from ragkit.cli import train_cli
from ragkit.rag import run_rag
from ragkit.config import get_settings
from ragkit.embeddings import create_embedding_fn, format_queries
from ragkit.models import generate_text
from ragkit.retrieval import DocumentStore, retrieve

app = cyclopts.App(name="ragkit", help="ragkit: 임베딩 검색 기반 RAG CLI")

# DS용 학습·평가 명령
for _command in (train_cli.split, train_cli.train, train_cli.evaluate, train_cli.compare):
    app.command(_command)


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


@app.command(name="index")
def index_command(
    corpus: Path | None = None,
    model: str | None = None,
    checkpoint: Path | None = None,
    backend: str | None = None,
    out: Path | None = None,
    device: str | None = None,
    batch_size: int = 64,
    sort_by_length: bool | None = None,
    threads: int | None = None,
) -> None:
    """코퍼스를 임베딩해 SQLite(sqlite-vec) 인덱스 파일을 만든다.

    같은 모델·같은 코퍼스의 파일이 이미 있으면 다시 만들지 않는다.

    Args:
        corpus: 코퍼스 JSON. 기본값 data/processed/law_docs.json.
        model: 임베딩 모델 이름. 기본값 Settings.embedding_model_name.
        checkpoint: 파인튜닝한 모델 폴더. 주면 이 폴더 이름이 모델 키가 된다.
        backend: onnx | torch. 기본값 Settings.embedding_backend.
        out: 인덱스 파일. 기본값 data/processed/index/<모델 키>.sqlite.
        device: torch 장치 auto | cuda | mps | cpu. 기본값 Settings.embedding_device.
        batch_size: 임베딩 배치 크기.
        sort_by_length: 길이순 배치 (--no-sort-by-length로 끔). 기본값 Settings 값.
        threads: CPU 스레드 수. 기본값 라이브러리 기본값.
    """
    from ragkit.data import load_corpus

    from ragkit.embeddings import get_profile
    from ragkit.retrieval import build_index, default_index_path, model_key

    settings = get_settings()
    corpus = corpus or settings.data_dir / "processed" / "law_docs.json"
    model = model or settings.embedding_model_name
    key = model_key(model, checkpoint)
    out = out or default_index_path(key)
    profile = get_profile(checkpoint or model)

    docs = load_corpus(corpus)
    embed_fn = create_embedding_fn(
        model,
        checkpoint_path=checkpoint,
        device=device,
        backend=backend,
        sort_by_length=sort_by_length,
        num_threads=threads,
    )
    index = build_index(
        docs, embed_fn, out, model_key=key, format_doc=profile.format_doc, batch_size=batch_size
    )
    print(f"인덱스: {out} (문서 {len(index):,}개, 모델 {index.model_key})")


@app.command(name="export-onnx")
def export_onnx_command(
    model_dir: Path,
    out_dir: Path | None = None,
) -> None:
    """임베딩 모델 폴더를 ONNX로 변환한다 (extra [train] 필요).

    Args:
        model_dir: config.json과 가중치가 있는 모델 폴더 (base 또는 파인튜닝 결과).
        out_dir: 출력 폴더. 없으면 model_dir/onnx/model.onnx에 쓴다.
    """
    from ragkit.models.onnx_export import export_onnx

    onnx_path = export_onnx(model_dir, out_dir)
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"ONNX 변환 완료: {onnx_path} ({size_mb:.1f} MB)")


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
