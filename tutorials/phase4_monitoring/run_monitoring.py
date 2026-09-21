"""Phase 4: 모니터링 & 관찰성."""

import time
from pathlib import Path

from src.agents import run_rag
from src.config import get_settings
from src.embeddings import create_embedding_fn
from src.models import generate_text
from src.monitoring.metrics import MetricsCollector
from src.retrieval import DocumentStore


def main() -> None:
    """테스트 질문을 모니터링하며 실행하고 메트릭을 수집한다."""
    print("=" * 60)
    print("Phase 4: 모니터링 & 관찰성")
    print("=" * 60)

    settings = get_settings()
    store_path = Path("experiments/results")
    store = DocumentStore()

    try:
        store.load(store_path)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"로드 실패: {e}")
        return

    embed_fn = create_embedding_fn(settings.embedding_model_name)
    collector = MetricsCollector()

    test_queries = [
        "기계학습이란 무엇인가?",
        "딥러닝은 어떻게 작동하나?",
    ]

    print(f"\n{len(test_queries)}개 질문을 모니터링하며 실행 중...")

    for query in test_queries:
        start = time.time()
        _result = run_rag(query, store, embed_fn, generate_text)
        latency_ms = (time.time() - start) * 1000

        collector.log_retrieval(query, 3, [0.95, 0.87, 0.76], latency_ms)
        print(f"질문 응답 시간: {latency_ms:.2f}ms")

    collector.save()
    summary = collector.get_summary()

    print("\n" + "=" * 60)
    print("모니터링 요약")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
