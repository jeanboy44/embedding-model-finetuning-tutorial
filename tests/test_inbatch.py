"""in-batch 점수표 · MNRL 손실 · 여유 테스트."""

import numpy as np

from ragkit.training.inbatch import batch_margins, in_batch_scores


def test_in_batch_scores_puts_positives_on_diagonal(questions, corpus_by_id) -> None:
    """정답과 같은 벡터를 주는 임베딩이면 대각선이 1이고 손실이 작다."""
    batch = [q for q in questions if q["hard_negative_ids"]][:2]
    targets = {
        corpus_by_id[q["positive_id"]]["title"]: np.eye(4)[i]
        for i, q in enumerate(batch)
    }

    def embed(texts: list[str]) -> np.ndarray:
        rows = []
        for i, text in enumerate(texts):
            if text.startswith("query: "):
                rows.append(np.eye(4)[i])
            else:
                hit = next((v for title, v in targets.items() if title in text), None)
                rows.append(hit if hit is not None else np.eye(4)[3])
        return np.array(rows)

    scores, loss = in_batch_scores(embed, batch, corpus_by_id)

    assert scores.shape == (2, 4)
    assert np.allclose(np.diag(scores), 1.0)
    assert loss < 0.01


def test_batch_margins_ignores_own_positive() -> None:
    scores = np.array([[0.9, 0.2, 0.5, 0.1], [0.3, 0.4, 0.2, 0.6]])

    assert np.allclose(batch_margins(scores), [0.4, -0.2])
