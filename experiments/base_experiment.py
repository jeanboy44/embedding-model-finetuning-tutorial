"""실험 베이스 클래스."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseExperiment(ABC):
    """모든 실험의 베이스 클래스.

    Args:
        name: 실험 이름.
        output_dir: 결과 저장 디렉토리.
    """

    def __init__(self, name: str, output_dir: Path) -> None:
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Any] = {}

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """실험을 실행한다.

        Returns:
            실험 결과 딕셔너리.
        """
        ...

    def evaluate_retrieval(
        self,
        retrieved_docs: list[str],
        relevant_docs: list[str],
    ) -> dict[str, float]:
        """검색 품질을 평가한다.

        Args:
            retrieved_docs: 검색된 문서 리스트.
            relevant_docs: 정답 문서 리스트.

        Returns:
            precision, recall, f1을 포함하는 딕셔너리.
        """
        matches = len(set(retrieved_docs) & set(relevant_docs))
        precision = matches / len(retrieved_docs) if retrieved_docs else 0
        recall = matches / len(relevant_docs) if relevant_docs else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * (precision * recall) / (precision + recall + 1e-10),
        }

    def save_results(self) -> None:
        """실험 결과를 JSON 파일로 저장한다."""
        results_file = self.output_dir / "results.json"
        results_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in self.results.items()
        }
        with results_file.open("w") as f:
            json.dump(results_serializable, f, indent=2)
