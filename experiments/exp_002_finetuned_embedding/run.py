"""실험 002: 파인튜닝 임베딩 모델 평가."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.base_experiment import BaseExperiment
from ragkit.embeddings import create_embedding_fn, format_passages, format_queries
from ragkit.retrieval import DocumentStore, retrieve

TEST_DOCS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand text.",
    "Computer vision enables machines to interpret images.",
    "Reinforcement learning uses rewards to train agents.",
]


class FinetuneEmbeddingExperiment(BaseExperiment):
    """파인튜닝 임베딩 실험.

    Args:
        config_path: 실험 설정 YAML 경로.
        embed_fn: 임베딩 생성 함수. None이면 config에서 모델 로드.
    """

    def __init__(
        self,
        config_path: Path,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        super().__init__("finetuned_embedding", Path(__file__).parent)
        with config_path.open() as f:
            self.config: dict[str, Any] = yaml.safe_load(f)

        if embed_fn is not None:
            self.embed_fn = embed_fn
        else:
            checkpoint_path = Path(self.config["checkpoint_path"])
            if not checkpoint_path.exists():
                print(
                    f"Note: Checkpoint {checkpoint_path} not found. Using base model."
                )
                checkpoint_path = None
            self.embed_fn = create_embedding_fn(
                self.config["model_name"], checkpoint_path=checkpoint_path
            )

    def run(self) -> dict[str, Any]:
        """파인튜닝 임베딩 실험을 실행한다.

        Returns:
            모델명, 파인튜닝 여부, 검색 결과를 포함하는 딕셔너리.
        """
        print("Running Fine-Tuned Embedding Experiment...")

        embeddings = self.embed_fn(format_passages(TEST_DOCS))

        store = DocumentStore()
        store.add_documents(TEST_DOCS, embeddings)

        test_query = "What is machine learning?"
        query_embedding = self.embed_fn(format_queries([test_query]))
        results = retrieve(store, query_embedding, k=3)

        self.results = {
            "model": self.config["model_name"],
            "finetuned": "checkpoint_path" in self.config,
            "num_documents": len(TEST_DOCS),
            "embedding_dim": embeddings.shape[1],
            "test_query": test_query,
            "top_3_results": [doc for doc, _ in results],
            "similarity_scores": [float(score) for _, score in results],
        }

        self.save_results()
        return self.results


if __name__ == "__main__":
    config = Path(__file__).parent / "config.yaml"
    FinetuneEmbeddingExperiment(config).run()
