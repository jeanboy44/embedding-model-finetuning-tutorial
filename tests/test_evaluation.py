"""검색 평가 지표 테스트."""

import math

import numpy as np
import pytest

from ragkit.data import doc_text
from ragkit.evaluation import (
    evaluate_index,
    evaluate_retrieval,
    first_rank,
    question_metrics,
)


def test_first_rank_counts_first_occurrence() -> None:
    assert first_rank(["a", "b", "a", "c"], "c") == 3
    assert first_rank(["a", "b"], "z") is None


def test_question_metrics() -> None:
    assert question_metrics(3, (1, 5)) == {
        "recall@1": 0.0,
        "recall@5": 1.0,
        "mrr@10": pytest.approx(1 / 3),
        "ndcg@10": pytest.approx(1 / math.log2(4)),
    }
    assert question_metrics(None, (1,)) == {"recall@1": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}
    assert question_metrics(11, (1,))["mrr@10"] == 0.0


def _doc(doc_id: str, parent: str, theme: str) -> dict:
    return {"id": doc_id, "title": doc_id, "text": "본문", "parent_id": parent, "theme": theme}


CORPUS = [
    _doc("A_1", "A", "youth"),
    _doc("A_2", "A", "youth"),
    _doc("B", "B", "youth"),
    _doc("C", "C", "traffic"),
]
VECTORS = {
    "passage: " + doc_text(CORPUS[0]): [1, 0, 0, 0],
    "passage: " + doc_text(CORPUS[1]): [0, 1, 0, 0],
    "passage: " + doc_text(CORPUS[2]): [0, 0, 1, 0],
    "passage: " + doc_text(CORPUS[3]): [0, 0, 0, 1],
    "query: q1": [0.1, 0.9, 0.5, 0.0],
    "query: q2": [0.0, 0.0, 0.9, 0.4],
}
QUESTIONS = [
    {"query": "q1", "positive_id": "A_1", "query_type": "situation"},
    {"query": "q2", "positive_id": "C", "query_type": "keyword", "related_ids": ["B"]},
]


def fake_embed(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """VECTORS에 적은 벡터를 L2 정규화해 돌려준다 (build_index가 batch_size를 넘긴다)."""
    out = np.array([VECTORS[t] for t in texts], dtype=float)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_evaluate_retrieval_doc_article_and_related() -> None:
    """q1: 정답 A_1이 3위(doc)지만 같은 조 A_2가 1위라 article로는 1위.
    q2: related B가 1위지만 순위에서 빠져 정답 C가 1위.
    """
    result = evaluate_retrieval(fake_embed, CORPUS, QUESTIONS, ks=(1, 5))

    assert result["n"] == 2
    assert result["doc"] == {
        "recall@1": 0.5,
        "recall@5": 1.0,
        "mrr@10": pytest.approx((1 / 3 + 1) / 2),
        "ndcg@10": pytest.approx((0.5 + 1) / 2),
    }
    assert result["article"]["recall@1"] == 1.0
    assert result["by_query_type"]["situation"]["doc"]["recall@1"] == 0.0
    assert result["by_theme"]["traffic"]["n"] == 1
    assert [(r["doc_rank"], r["article_rank"]) for r in result["per_question"]] == [(3, 1), (1, 1)]


def test_evaluate_index_matches_numpy(tmp_path) -> None:
    """SQLite 인덱스로 평가해도 numpy 평가와 결과가 같다."""
    from ragkit.retrieval import build_index

    index = build_index(CORPUS, fake_embed, tmp_path / "t.sqlite", model_key="fake")

    by_index = evaluate_index(index, fake_embed, CORPUS, QUESTIONS, ks=(1, 5))
    by_numpy = evaluate_retrieval(fake_embed, CORPUS, QUESTIONS, ks=(1, 5))

    assert by_index["per_question"] == by_numpy["per_question"]
    assert by_index["doc"] == pytest.approx(by_numpy["doc"])
    assert by_index["article"] == pytest.approx(by_numpy["article"])


def test_evaluate_uses_model_formatters(tmp_path) -> None:
    """모델별 입력 형식(format_query / format_doc)을 주면 앞 문구 대신 그것으로 임베딩한다."""
    from ragkit.retrieval import build_index

    seen: list[str] = []

    def embed(texts: list[str], batch_size: int | None = None) -> np.ndarray:
        seen.extend(texts)
        return np.array([[1.0, 0.0] if t.startswith("Q|") else [0.9, 0.1] for t in texts])

    def fq(q: str) -> str:
        return f"Q|{q}"

    def fd(d: dict) -> str:
        return f"D|{d['title']}"

    corpus = [_doc("A", "A", "t"), _doc("B", "B", "t")]
    questions = [{"query": "q", "positive_id": "A"}]

    evaluate_retrieval(embed, corpus, questions, format_query=fq, format_doc=fd)
    assert seen == ["D|A", "D|B", "Q|q"]

    seen.clear()
    index = build_index(corpus, embed, tmp_path / "f.sqlite", model_key="f", format_doc=fd)
    seen.clear()
    evaluate_index(index, embed, corpus, questions, format_query=fq)
    assert seen == ["Q|q"]


def test_evaluate_retrieval_reuses_corpus_embeddings() -> None:
    """corpus_embeddings를 주면 코퍼스를 다시 임베딩하지 않는다."""
    corpus = [_doc("A", "A", "t"), _doc("B", "B", "t")]
    seen: list[str] = []

    def embed(texts: list[str]) -> np.ndarray:
        seen.extend(texts)
        return np.array([[1.0, 0.0]] * len(texts))

    evaluate_retrieval(
        embed,
        corpus,
        [{"query": "q", "positive_id": "A"}],
        corpus_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    assert seen == ["query: q"]


def test_evaluate_retrieval_rejects_bad_input() -> None:
    corpus = [_doc("A", "A", "t")]
    embed = lambda texts: np.ones((len(texts), 1))
    with pytest.raises(ValueError, match="평가할 질문"):
        evaluate_retrieval(embed, corpus, [])
    with pytest.raises(ValueError, match="코퍼스에 없는"):
        evaluate_retrieval(embed, corpus, [{"query": "q", "positive_id": "Z"}])
