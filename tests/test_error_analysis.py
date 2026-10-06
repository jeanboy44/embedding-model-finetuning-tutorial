"""오답 분석 도우미 테스트 (ragkit.evaluation)."""

import pytest

from ragkit.evaluation import (
    RANK_BUCKETS,
    RELATIONS,
    compare_hits,
    confusion_relation,
    query_overlap,
    rank_bucket,
)


def test_rank_bucket() -> None:
    assert [rank_bucket(r) for r in (1, 5, 6, 10, 11, 30, 31, 100, None)] == [
        "1~5위",
        "1~5위",
        "6~10위",
        "6~10위",
        "11~30위",
        "11~30위",
        "31~100위",
        "31~100위",
        "100위 밖",
    ]
    assert rank_bucket(101) == RANK_BUCKETS[-1]


def _doc(
    doc_id: str, category: str, law_type: str, theme: str, parent: str | None = None
) -> dict:
    return {
        "id": doc_id,
        "category": category,
        "law_type": law_type,
        "theme": theme,
        "parent_id": parent,
    }


def test_confusion_relation() -> None:
    positive = _doc("A_법률_제1조_제1항", "A", "법률", "tax", parent="A_법률_제1조")
    cases = {
        "same_article": _doc(
            "A_법률_제1조_제2항", "A", "법률", "tax", parent="A_법률_제1조"
        ),
        "same_law_same_type": _doc("A_법률_제2조", "A", "법률", "tax"),
        "same_law_other_type": _doc("A_시행령_제1조", "A", "시행령", "tax"),
        "other_law_same_theme": _doc("B_법률_제1조", "B", "법률", "tax"),
        "other_law_other_theme": _doc("C_법률_제1조", "C", "법률", "youth"),
    }
    for expected, wrong in cases.items():
        assert confusion_relation(positive, wrong) == expected
    assert set(cases) == set(RELATIONS)


def test_query_overlap() -> None:
    assert query_overlap("주휴수당", "주휴수당을 지급한다") == 1.0
    assert query_overlap("알바", "근로자") == 0.0
    assert query_overlap("주휴 알바", "주휴일") == pytest.approx(
        1 / 3
    )  # 주휴·휴알·알바 중 주휴만
    assert query_overlap("", "본문") == 0.0


def _result(ranks: dict[str, int | None]) -> dict:
    return {
        "per_question": [
            {
                "qid": qid,
                "query": qid,
                "positive_id": "p",
                "law": "L",
                "doc_rank": rank,
                "top10": [],
            }
            for qid, rank in ranks.items()
        ]
    }


def test_compare_hits_groups_questions() -> None:
    base = _result({"a": 1, "b": 7, "c": None, "d": 3})
    other = _result({"a": 2, "b": 4, "c": 50, "d": None})
    groups = compare_hits(base, other, k=5)
    assert {name: [row["qid"] for row in rows] for name, rows in groups.items()} == {
        "fixed": ["b"],
        "broken": ["d"],
        "both_wrong": ["c"],
        "both_right": ["a"],
    }
    assert (
        groups["fixed"][0]["base_rank"] == 7 and groups["fixed"][0]["other_rank"] == 4
    )


def test_compare_hits_rejects_different_question_sets() -> None:
    with pytest.raises(ValueError):
        compare_hits(_result({"a": 1}), _result({"b": 1}))
