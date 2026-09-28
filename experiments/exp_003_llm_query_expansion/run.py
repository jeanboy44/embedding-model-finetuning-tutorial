"""실험 003: LLM 쿼리 확장을 통한 검색 개선."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.base_experiment import BaseExperiment
from ragkit.embeddings import create_embedding_fn, format_passages, format_queries
from ragkit.models import generate_text
from ragkit.retrieval import DocumentStore, retrieve

TEST_DOCS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand text.",
    "Computer vision enables machines to interpret images.",
    "Reinforcement learning uses rewards to train agents.",
]


class LLMQueryExpansionExperiment(BaseExperiment):
    """LLM 쿼리 확장 실험.

    Args:
        config_path: 실험 설정 YAML 경로.
        embed_fn: 임베딩 생성 함수. None이면 config에서 모델 로드.
        generate_fn: 텍스트 생성 함수. None이면 Gemini 기본 사용.
    """

    def __init__(
        self,
        config_path: Path,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        generate_fn: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__("llm_query_expansion", Path(__file__).parent)
        with config_path.open() as f:
            self.config: dict[str, Any] = yaml.safe_load(f)

        self.embed_fn = embed_fn or create_embedding_fn(self.config["model_name"])
        self.generate_fn = generate_fn or (
            lambda prompt: generate_text(prompt, model_name=self.config["llm_model"])
        )

    def run(self) -> dict[str, Any]:
        """LLM 쿼리 확장 실험을 실행한다.

        Returns:
            모델명, 확장 쿼리, 검색 결과를 포함하는 딕셔너리.
        """
        print("Running LLM Query Expansion Experiment...")

        embeddings = self.embed_fn(format_passages(TEST_DOCS))
        store = DocumentStore()
        store.add_documents(TEST_DOCS, embeddings)

        original_query = "What is machine learning?"

        expansion_prompt = (
            f"Expand this query into 3 variations to improve retrieval:\n"
            f'"{original_query}"\n'
            f"Return only the variations, one per line."
        )

        expanded_text = self.generate_fn(expansion_prompt)
        expanded_queries = [q.strip() for q in expanded_text.split("\n") if q.strip()]

        all_queries = [original_query] + expanded_queries[:3]
        query_embeddings = self.embed_fn(format_queries(all_queries))

        all_results: list[list[str]] = []
        for i, _query in enumerate(all_queries):
            results = retrieve(store, query_embeddings[i : i + 1], k=3)
            all_results.append([doc for doc, _ in results])

        self.results = {
            "model": self.config["model_name"],
            "llm_used": self.config["llm_model"],
            "original_query": original_query,
            "expanded_queries": all_queries,
            "retrieval_results": all_results,
            "method": "query_expansion",
        }

        self.save_results()
        return self.results


if __name__ == "__main__":
    config = Path(__file__).parent / "config.yaml"
    LLMQueryExpansionExperiment(config).run()
