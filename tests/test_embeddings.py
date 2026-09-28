"""임베딩 풀링 및 앞 문구 테스트."""

import torch

from ragkit.embeddings import format_passages, format_queries, mean_pool


def test_mean_pool_ignores_padding() -> None:
    """패딩 토큰 값은 평균에 포함되지 않는다."""
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = mean_pool(hidden, mask)

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_mean_pool_same_sentence_with_and_without_padding() -> None:
    """같은 문장은 배치 내 패딩 길이와 무관하게 같은 임베딩을 갖는다."""
    tokens = torch.randn(1, 3, 4)
    padded = torch.cat([tokens, torch.randn(1, 2, 4)], dim=1)

    short = mean_pool(tokens, torch.ones(1, 3, dtype=torch.long))
    long = mean_pool(padded, torch.tensor([[1, 1, 1, 0, 0]]))

    assert torch.allclose(short, long)


def test_format_prefixes() -> None:
    """기본 설정(e5)의 query/passage 앞 문구가 붙는다."""
    assert format_queries(["질문"]) == ["query: 질문"]
    assert format_passages(["문서"]) == ["passage: 문서"]
