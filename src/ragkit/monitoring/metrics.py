"""메트릭 수집."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class RetrievalMetrics:
    """검색 메트릭.

    Attributes:
        query: 검색 쿼리.
        num_results: 반환 문서 수.
        top_scores: 상위 유사도 점수.
        latency_ms: 지연시간 (밀리초).
        timestamp: ISO 형식 타임스탬프.
    """

    query: str
    num_results: int
    top_scores: list[float]
    latency_ms: float
    timestamp: str


@dataclass
class GenerationMetrics:
    """생성 메트릭.

    Attributes:
        prompt_tokens: 입력 토큰 수.
        output_tokens: 출력 토큰 수.
        latency_ms: 지연시간 (밀리초).
        timestamp: ISO 형식 타임스탬프.
    """

    prompt_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: str


class MetricsCollector:
    """메트릭을 수집하고 저장한다.

    Args:
        output_dir: 메트릭 저장 디렉토리.

    Example:
        >>> collector = MetricsCollector(Path("logs/metrics"))
        >>> collector.log_retrieval("query", 3, [0.95, 0.87], 12.5)
        >>> collector.save()
    """

    def __init__(self, output_dir: Path = Path("logs/metrics")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_metrics: list[RetrievalMetrics] = []
        self.generation_metrics: list[GenerationMetrics] = []

    def log_retrieval(
        self,
        query: str,
        num_results: int,
        top_scores: list[float],
        latency_ms: float,
    ) -> None:
        """검색 메트릭을 기록한다.

        Args:
            query: 검색 쿼리.
            num_results: 반환 문서 수.
            top_scores: 상위 유사도 점수.
            latency_ms: 지연시간 (밀리초).
        """
        self.retrieval_metrics.append(
            RetrievalMetrics(
                query=query,
                num_results=num_results,
                top_scores=top_scores,
                latency_ms=latency_ms,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

    def log_generation(
        self,
        prompt_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """생성 메트릭을 기록한다.

        Args:
            prompt_tokens: 입력 토큰 수.
            output_tokens: 출력 토큰 수.
            latency_ms: 지연시간 (밀리초).
        """
        self.generation_metrics.append(
            GenerationMetrics(
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

    def get_summary(self) -> dict:
        """메트릭 요약을 반환한다.

        Returns:
            검색/생성 횟수와 평균 지연시간을 포함하는 딕셔너리.
        """
        avg_retrieval = (
            sum(m.latency_ms for m in self.retrieval_metrics)
            / len(self.retrieval_metrics)
            if self.retrieval_metrics
            else 0
        )
        avg_generation = (
            sum(m.latency_ms for m in self.generation_metrics)
            / len(self.generation_metrics)
            if self.generation_metrics
            else 0
        )

        return {
            "retrieval_count": len(self.retrieval_metrics),
            "generation_count": len(self.generation_metrics),
            "avg_retrieval_latency_ms": avg_retrieval,
            "avg_generation_latency_ms": avg_generation,
        }

    def save(self) -> None:
        """메트릭을 JSON 파일로 저장한다."""
        metrics = {
            "retrieval": [asdict(m) for m in self.retrieval_metrics],
            "generation": [asdict(m) for m in self.generation_metrics],
            "summary": self.get_summary(),
        }

        output_file = (
            self.output_dir
            / f"metrics_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        )
        with output_file.open("w") as f:
            json.dump(metrics, f, indent=2)
