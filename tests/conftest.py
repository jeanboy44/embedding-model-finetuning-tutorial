"""공용 테스트 fixture: 가짜 법령 코퍼스와 질문."""

import pytest

# 테마별 법령. youth는 법령이 4개, traffic·electric은 3개 미만이라 분할 때 한 그룹으로 합쳐진다.
LAWS = {
    "youth": ["가법", "나법", "다법", "라법"],
    "traffic": ["마법"],
    "electric": ["바법", "사법"],
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: 실제 모델을 쓰는 느린 테스트")


def make_doc(law: str, theme: str, article: str, paragraph: str = "") -> dict:
    """prepare_law_data.py 출력과 같은 필드를 가진 가짜 문서."""
    parent = f"{law}_법률_{article}"
    doc_id = f"{parent}_{paragraph}" if paragraph else parent
    return {
        "id": doc_id,
        "title": f"{law} {article} {paragraph}".strip(),
        "text": f"{law} {article} {paragraph} 본문".replace("  ", " "),
        "category": law,
        "theme": theme,
        "law_name": law,
        "parent_id": parent,
        "paragraph": paragraph,
    }


@pytest.fixture
def corpus() -> list[dict]:
    """법령마다 문서 4개: 제1조는 두 조각(제1항, 제2항), 제2조, 제3조."""
    docs = []
    for theme, laws in LAWS.items():
        for law in laws:
            docs.append(make_doc(law, theme, "제1조", "제1항"))
            docs.append(make_doc(law, theme, "제1조", "제2항"))
            docs.append(make_doc(law, theme, "제2조"))
            docs.append(make_doc(law, theme, "제3조"))
    return docs


@pytest.fixture
def corpus_by_id(corpus: list[dict]) -> dict[str, dict]:
    return {doc["id"]: doc for doc in corpus}


@pytest.fixture
def questions(corpus: list[dict]) -> list[dict]:
    """문서마다 질문 1개. hard negative는 같은 법령의 다른 조 문서 2개."""
    rows = []
    for doc in corpus:
        others = [
            d["id"]
            for d in corpus
            if d["category"] == doc["category"] and d["parent_id"] != doc["parent_id"]
        ]
        rows.append(
            {
                "query": f"{doc['id']} 에 대한 질문",
                "positive_id": doc["id"],
                "hard_negative_ids": others[:2],
                "query_type": "situation",
                "answer": "답",
                "related_ids": [],
            }
        )
    return rows
