"""ragkit.data 테스트."""

import json

from ragkit.data import (
    doc_text,
    filter_questions,
    load_corpus,
    load_questions,
    relevance_key,
)


def test_relevance_key_uses_parent_id(corpus_by_id) -> None:
    """조각 문서는 원래 조 id, parent_id가 없으면 자기 id."""
    assert relevance_key(corpus_by_id["가법_법률_제1조_제2항"]) == "가법_법률_제1조"
    assert relevance_key({"id": "x"}) == "x"


def test_doc_text_joins_title_and_text(corpus_by_id) -> None:
    doc = corpus_by_id["가법_법률_제2조"]
    assert doc_text(doc) == "가법 제2조\n가법 제2조 본문"


def test_load_corpus_and_questions(tmp_path, corpus, questions) -> None:
    """파일 하나 또는 폴더(안의 *.jsonl 이름순)를 읽고, 빈 줄은 건너뛴다."""
    corpus_path = tmp_path / "law_docs.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    assert load_corpus(corpus_path) == corpus

    qdir = tmp_path / "questions"
    qdir.mkdir()
    lines = [json.dumps(q, ensure_ascii=False) for q in questions]
    (qdir / "b__p01.jsonl").write_text("\n".join(lines[2:4]) + "\n\n", encoding="utf-8")
    (qdir / "a__p01.jsonl").write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    assert load_questions(qdir) == questions[:4]
    assert load_questions(qdir / "a__p01.jsonl") == questions[:2]


def test_filter_questions_drops_missing_and_bad_negatives(corpus_by_id) -> None:
    """없는 positive는 질문째 빼고, negative에서는 없는 id·related·정답 자신·중복을 지운다.

    같은 조의 다른 조각(예: 제6조 제1항 질문의 제6조 제2항)은 다른 질문에 답하는 진짜
    hard negative라 남긴다.
    """
    questions = [
        {"query": "없는 정답", "positive_id": "없는법_법률_제1조", "hard_negative_ids": []},
        {
            "query": "제1항 질문",
            "positive_id": "가법_법률_제1조_제1항",
            "hard_negative_ids": [
                "가법_법률_제1조_제2항",  # 같은 조의 다른 조각 → 남김
                "가법_법률_제1조_제1항",  # 정답 자신 → 지움
                "가법_법률_제2조",  # related → 지움
                "없는법_법률_제9조",  # 코퍼스에 없음 → 지움
                "가법_법률_제3조",  # 남김
                "가법_법률_제3조",  # 중복 → 하나만 남김
            ],
            "related_ids": ["가법_법률_제2조"],
        },
    ]

    kept, stats = filter_questions(questions, corpus_by_id)

    assert [q["query"] for q in kept] == ["제1항 질문"]
    assert kept[0]["hard_negative_ids"] == ["가법_법률_제1조_제2항", "가법_법률_제3조"]
    assert stats == {"missing_positive": 1, "dropped_negatives": 4}
    # 입력은 바꾸지 않는다
    assert len(questions[1]["hard_negative_ids"]) == 6
