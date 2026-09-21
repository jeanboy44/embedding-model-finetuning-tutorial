"""Phase 3: RAG 에이전트 개발."""

from pathlib import Path

from src.agents import run_rag
from src.config import get_settings
from src.embeddings import create_embedding_fn
from src.models import generate_text
from src.retrieval import DocumentStore


def main() -> None:
    """RAG 에이전트를 초기화하고 대화형으로 질문에 답변한다."""
    print("=" * 60)
    print("Phase 3: RAG 에이전트 개발")
    print("=" * 60)

    settings = get_settings()
    store_path = Path("experiments/results")
    store = DocumentStore()

    try:
        store.load(store_path)
        print(f"  {len(store.documents)}개의 문서 로드됨")
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"  로드 실패: {e}")
        return

    embed_fn = create_embedding_fn(settings.embedding_model_name)
    print("  RAG 에이전트 초기화됨")
    print("\n질문을 입력하세요 (Ctrl+C로 종료):")

    try:
        while True:
            query = input("\n질문: ").strip()
            if not query:
                continue

            result = run_rag(query, store, embed_fn, generate_text)
            print(f"\n답변: {result['answer'][:200]}...")

    except KeyboardInterrupt:
        print("\n에이전트 종료됨.")


if __name__ == "__main__":
    main()
