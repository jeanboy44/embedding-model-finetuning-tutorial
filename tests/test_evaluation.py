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

    # top10은 점수가 같은 문서의 순서가 구현마다 다를 수 있어 순위만 비교한다
    strip = lambda rows: [{k: v for k, v in r.items() if k != "top10"} for r in rows]
    assert strip(by_index["per_question"]) == strip(by_numpy["per_question"])
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


def test_evaluate_multi_counts_alt_positives_and_groups_by_law() -> None:
    """multi 판정: 정답 또는 alt_positive_ids 중 가장 먼저 나온 문서의 순위.
    q1: 정답 A_1은 3위지만 대체 정답 B가 2위 → multi 2위. q2: 대체 정답이 없으면 doc과 같다.
    by_law: positive 문서의 법령(category)별 지표.
    """
    corpus = [{**doc, "category": "갑법" if doc["id"].startswith("A") else "을법"} for doc in CORPUS]
    questions = [{**QUESTIONS[0], "alt_positive_ids": ["B"]}, QUESTIONS[1]]

    result = evaluate_retrieval(fake_embed, corpus, questions, ks=(1, 5))

    assert [r["multi_rank"] for r in result["per_question"]] == [2, 1]
    # partial_ids는 multi 판정에서만 순위에서 뺀다: A_2를 빼면 multi에서 B가 1위, doc은 그대로 3위
    partial = [{**questions[0], "partial_ids": ["A_2"]}, QUESTIONS[1]]
    result_p = evaluate_retrieval(fake_embed, corpus, partial, ks=(1, 5))
    assert [r["multi_rank"] for r in result_p["per_question"]] == [1, 1]
    assert [r["doc_rank"] for r in result_p["per_question"]] == [3, 1]
    assert result["multi"]["recall@1"] == 0.5 and result["multi"]["mrr@10"] == pytest.approx((1 / 2 + 1) / 2)
    assert result["doc"]["recall@1"] == 0.5  # doc 판정은 그대로
    assert set(result["by_law"]) == {"갑법", "을법"}
    assert result["by_law"]["갑법"]["n"] == 1


def test_per_question_has_qid_law_and_top10() -> None:
    """질문별 결과에 qid(판정 파일·다른 run과 잇는 키), 법령, 상위 10개 id가 있다."""
    from ragkit.data import question_key

    corpus = [{**doc, "category": "갑법"} for doc in CORPUS]
    result = evaluate_retrieval(fake_embed, corpus, QUESTIONS, ks=(1, 5))

    row = result["per_question"][0]
    assert row["qid"] == question_key(QUESTIONS[0])
    assert row["law"] == "갑법"
    assert row["top10"][:3] == ["A_2", "B", "A_1"]


def test_paired_test_reports_delta_ci_mcnemar_and_law_deltas() -> None:
    """같은 질문 집합의 두 결과를 질문 단위로 짝지어 Δ, bootstrap CI, McNemar p, 법령별 Δ를 낸다."""
    from ragkit.evaluation import paired_test

    def result(ranks):
        return {"per_question": [
            {"qid": f"q{i}", "law": "갑법" if i < 10 else "을법", "multi_rank": r} for i, r in enumerate(ranks)
        ]}

    base = result([1] * 10 + [9] * 10)  # 갑법 10개 적중, 을법 10개 실패
    other = result([1] * 10 + [2] * 6 + [9] * 4)  # 을법 6개를 새로 맞힘

    out = paired_test(base, other, k=5, n_boot=2000, seed=0)

    assert out["n"] == 20 and out["base"] == 0.5 and out["other"] == 0.8
    assert out["delta"] == pytest.approx(0.3)
    assert out["ci95"][0] > 0
    assert out["fixed"] == 6 and out["broken"] == 0
    assert out["mcnemar_p"] < 0.05
    assert out["by_law"]["을법"]["delta"] == pytest.approx(0.6)
    assert out["by_law"]["갑법"]["delta"] == 0.0


def test_paired_test_rejects_different_question_sets() -> None:
    from ragkit.evaluation import paired_test

    a = {"per_question": [{"qid": "q1", "law": "x", "multi_rank": 1}]}
    b = {"per_question": [{"qid": "q2", "law": "x", "multi_rank": 1}]}
    with pytest.raises(ValueError):
        paired_test(a, b)
