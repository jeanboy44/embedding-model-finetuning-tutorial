"""MNRL이 배치 하나를 어떻게 보는지 숫자로 보여 주는 도구 (강의 5 · 6교시 파일럿).

질문 n개 × (배치의 정답 n개 + 질문마다 hard negative n개) 코사인 점수표를 만들고,
MultipleNegativesRankingLoss와 같은 방식(scale을 곱한 softmax, 대각선이 정답)으로 손실을 잰다.
학습 자체는 sentence-transformers가 하고, 여기서는 학습 전후에 같은 배치를 다시 재는 데만 쓴다.
"""

from collections.abc import Callable, Sequence

import numpy as np

from ragkit.data import doc_text
from ragkit.embeddings import format_passages, format_queries

SCALE = 20.0  # MNRL이 코사인 유사도에 곱하는 값 (sentence-transformers 기본)


def in_batch_scores(
    embed: Callable[[list[str]], np.ndarray],
    batch: Sequence[dict],
    corpus_by_id: dict[str, dict],
    *,
    scale: float = SCALE,
) -> tuple[np.ndarray, float]:
    """배치의 점수표와 MNRL 손실(평균)을 돌려준다.

    Args:
        embed: 문장 목록 → 정규화된 임베딩 행렬.
        batch: 질문 목록. 질문마다 positive_id와 hard_negative_ids(1개 이상)가 있어야 한다.
        corpus_by_id: id → 문서.
        scale: 코사인 유사도에 곱하는 값.

    Returns:
        (점수표 n × 2n, 손실). 열은 정답 n개 다음 hard negative n개 순서이고, [i, i]가 질문 i의 정답이다.
    """
    queries = embed(format_queries([q["query"] for q in batch]))
    passages = [doc_text(corpus_by_id[q["positive_id"]]) for q in batch]
    passages += [doc_text(corpus_by_id[q["hard_negative_ids"][0]]) for q in batch]
    scores = queries @ embed(format_passages(passages)).T
    logits = scale * scores
    logits -= logits.max(axis=1, keepdims=True)
    log_softmax = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    loss = float(-np.mean([log_softmax[i, i] for i in range(len(batch))]))
    return scores, loss


def batch_margins(scores: np.ndarray) -> np.ndarray:
    """질문마다 정답 점수 − 가장 높은 오답 점수. 0보다 크면 그 줄에서 정답이 1위다."""
    others = scores.copy()
    np.fill_diagonal(others[:, : len(scores)], -np.inf)
    return np.diag(scores) - others.max(axis=1)
