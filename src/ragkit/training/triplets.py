"""질문 → 대조 학습 예시 (anchor, positive, negative_1..n).

MultipleNegativesRankingLoss는 같은 배치의 다른 positive를 자동으로 negative로 쓰고(in-batch negative),
negative_n 열의 hard negative를 추가 후보로 쓴다.
"""

from ragkit.data import doc_text


def to_examples(
    questions: list[dict],
    corpus_by_id: dict[str, dict],
    *,
    num_negatives: int = 1,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
) -> list[dict]:
    """질문을 학습 예시 행으로 바꾼다.

    모든 행의 열 수가 같아야 하므로 hard negative가 num_negatives보다 적으면 있는 것을 반복하고,
    하나도 없으면 그 질문을 건너뛴다.

    Args:
        questions: filter_questions를 거친 질문 목록.
        corpus_by_id: id → 문서.
        num_negatives: 행마다 넣을 hard negative 수.
        query_prefix: 질문 앞 문구 (e5: "query: ").
        passage_prefix: 문서 앞 문구 (e5: "passage: ").

    Returns:
        {"anchor", "positive", "negative_1", ...} 딕셔너리 목록.
    """
    rows: list[dict] = []
    for question in questions:
        negatives = question["hard_negative_ids"]
        if not negatives:
            continue
        row = {
            "anchor": query_prefix + question["query"],
            "positive": passage_prefix + doc_text(corpus_by_id[question["positive_id"]]),
        }
        for i in range(num_negatives):
            negative = corpus_by_id[negatives[i % len(negatives)]]
            row[f"negative_{i + 1}"] = passage_prefix + doc_text(negative)
        rows.append(row)
    return rows
