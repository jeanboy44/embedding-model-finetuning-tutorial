"""모니터링 로깅 함수 — loguru 기반."""

from loguru import logger


def log_query(query: str, embedding_shape: tuple[int, ...]) -> None:
    """검색 쿼리를 로깅한다.

    Args:
        query: 검색 쿼리 문자열.
        embedding_shape: 임베딩 배열의 shape.
    """
    logger.info(f"Query: {query}, embedding_shape: {embedding_shape}")


def log_retrieval_results(query: str, num_results: int, scores: list[float]) -> None:
    """검색 결과를 로깅한다.

    Args:
        query: 검색 쿼리 문자열.
        num_results: 반환된 문서 수.
        scores: 유사도 점수 리스트.
    """
    logger.info(f"Query: {query}, retrieved {num_results} docs, scores: {scores}")


def log_generation(prompt_len: int, output_tokens: int) -> None:
    """생성 결과를 로깅한다.

    Args:
        prompt_len: 입력 프롬프트 길이.
        output_tokens: 출력 토큰 수.
    """
    logger.info(f"Generated {output_tokens} tokens, prompt_len: {prompt_len}")


def log_retrieval_accuracy(accuracy: float, k: int) -> None:
    """검색 정확도를 로깅한다.

    Args:
        accuracy: 정확도 (0.0~1.0).
        k: top-k 값.
    """
    logger.info(f"Retrieval@{k}: {accuracy:.4f}")


def log_latency(operation: str, latency_ms: float) -> None:
    """작업 지연시간을 로깅한다.

    Args:
        operation: 작업 이름.
        latency_ms: 지연시간 (밀리초).
    """
    logger.info(f"{operation} latency: {latency_ms:.2f}ms")
