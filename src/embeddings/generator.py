"""임베딩 생성 함수."""

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from src.config import get_settings
from src.models import load_embedding_model, load_tokenizer


class Tokenizer(Protocol):
    """토크나이저 프로토콜."""

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]: ...


def format_queries(texts: list[str]) -> list[str]:
    """검색 쿼리에 모델이 요구하는 앞 문구를 붙인다 (e5: "query: ").

    Args:
        texts: 쿼리 텍스트 리스트.

    Returns:
        앞 문구가 붙은 텍스트 리스트.
    """
    prefix = get_settings().query_prefix
    return [f"{prefix}{t}" for t in texts]


def format_passages(texts: list[str]) -> list[str]:
    """검색 대상 문서에 모델이 요구하는 앞 문구를 붙인다 (e5: "passage: ").

    Args:
        texts: 문서 텍스트 리스트.

    Returns:
        앞 문구가 붙은 텍스트 리스트.
    """
    prefix = get_settings().passage_prefix
    return [f"{prefix}{t}" for t in texts]


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """패딩 토큰을 제외하고 토큰 임베딩의 평균을 구한다.

    Args:
        hidden_states: (batch, seq_len, dim) 형태의 마지막 은닉 상태.
        attention_mask: (batch, seq_len) 형태의 어텐션 마스크 (패딩=0).

    Returns:
        (batch, dim) 형태의 문장 임베딩.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def generate_embeddings(
    texts: list[str],
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    *,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """텍스트 리스트에 대한 임베딩을 생성한다.

    Args:
        texts: 임베딩할 텍스트 리스트.
        model: 임베딩 모델 (이미 로드 및 평가 모드 설정됨).
        tokenizer: 토크나이저 인스턴스.
        batch_size: 배치 크기.
        device: 디바이스 문자열. None이면 자동 감지.

    Returns:
        (N, embedding_dim) 형태의 L2 정규화된 임베딩 배열.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def create_embedding_fn(
    model_name: str,
    checkpoint_path: Path | None = None,
    device: str | None = None,
):
    """모델을 로드하고 임베딩 생성 함수를 반환한다.

    팩토리 함수로, 모델 로드를 한 번만 수행하고 이후 호출에서 재사용한다.

    Args:
        model_name: HuggingFace 모델 이름.
        checkpoint_path: 파인튜닝된 체크포인트 경로.
        device: 디바이스 문자열. None이면 자동 감지.

    Returns:
        texts를 받아 np.ndarray를 반환하는 함수.

    Example:
        >>> embed = create_embedding_fn("intfloat/multilingual-e5-small")
        >>> doc_embeddings = embed(format_passages(["머신러닝은 AI의 한 분야이다."]))
        >>> query_embedding = embed(format_queries(["머신러닝이란?"]))
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_embedding_model(model_name, checkpoint_path, device)
    tokenizer = load_tokenizer(model_name)

    def _embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
        return generate_embeddings(
            texts, model, tokenizer, batch_size=batch_size, device=device
        )

    return _embed
