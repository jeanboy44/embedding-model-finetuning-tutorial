"""학습 예시 변환 테스트."""

from ragkit.data import doc_text
from ragkit.training.triplets import to_examples


def test_to_examples_adds_prefixes(questions, corpus_by_id) -> None:
    question = questions[0]

    [row] = to_examples([question], corpus_by_id)

    positive = corpus_by_id[question["positive_id"]]
    negative = corpus_by_id[question["hard_negative_ids"][0]]
    assert row == {
        "anchor": "query: " + question["query"],
        "positive": "passage: " + doc_text(positive),
        "negative_1": "passage: " + doc_text(negative),
    }


def test_to_examples_repeats_short_negatives_and_skips_empty(corpus_by_id) -> None:
    """negative가 모자라면 있는 것을 반복하고, 0개면 그 질문을 건너뛴다."""
    questions = [
        {"query": "하나", "positive_id": "가법_법률_제2조", "hard_negative_ids": ["가법_법률_제3조"]},
        {"query": "없음", "positive_id": "가법_법률_제2조", "hard_negative_ids": []},
    ]

    rows = to_examples(questions, corpus_by_id, num_negatives=3, query_prefix="", passage_prefix="")

    assert len(rows) == 1
    expected = doc_text(corpus_by_id["가법_법률_제3조"])
    assert [rows[0][f"negative_{i}"] for i in (1, 2, 3)] == [expected] * 3
    assert rows[0]["anchor"] == "하나"
