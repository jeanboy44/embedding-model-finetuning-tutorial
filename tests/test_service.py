"""Searcher 테스트: 법령 필터 검색, 조문 조회, 스트리밍 답변 이벤트 (가짜 임베딩·가짜 LLM)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from ragkit.models import Generation
from ragkit.retrieval import build_index
from ragkit.service import DeltaEvent, DoneEvent, HitsEvent, Searcher


def _embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    vecs = np.array([[t.count("휴"), t.count("임"), 0.5] for t in texts], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _doc(doc_id: str, law: str, text: str, parent: str | None = None, theme: str = "youth") -> dict:
    return {"id": doc_id, "parent_id": parent or doc_id, "title": f"{doc_id} 제목", "text": text,
            "law_name": law, "law_type": "법률", "theme": theme}


DOCS = [
    _doc("근로_제55조", "근로기준법", "휴일 휴일 휴일"),
    _doc("근로_제56조_제1항", "근로기준법", "휴일 근로 임금", parent="근로_제56조"),
    _doc("근로_제56조_제2항", "근로기준법", "연장 근로", parent="근로_제56조"),
    _doc("임대_제4조", "주택임대차보호법", "임대차 임차인 임대인"),
    _doc("최저_제6조", "최저임금법", "임금 임금 휴", theme="tax"),
]


class _FakeStream:
    def __init__(self, pieces: list[str], fail: bool = False) -> None:
        self.pieces, self.fail = pieces, fail
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> Iterator[str | Generation]:
        self.prompts.append(prompt)
        yield from self.pieces
        if self.fail:
            raise RuntimeError("503 busy")
        yield Generation(text="".join(self.pieces), input_tokens=11, output_tokens=2)


@pytest.fixture
def index(tmp_path: Path):
    return build_index(DOCS, _embed, tmp_path / "idx.sqlite", model_key="fake")


def test_search_all_laws_or_only_given_laws(index) -> None:
    """laws가 없으면 전체, 있으면 그 법령들 안에서만 찾는다."""
    searcher = Searcher(index, _embed)

    assert searcher.search("임금", k=1)[0].id == "임대_제4조"
    assert searcher.search("임금", k=1, laws=[])[0].id == "임대_제4조"
    hits = searcher.search("임금", k=5, laws=["근로기준법", "최저임금법"])
    assert {h.metadata["law_name"] for h in hits} == {"근로기준법", "최저임금법"}
    assert hits[0].id == "최저_제6조"


def test_unknown_law_suggests_similar_names(index) -> None:
    """모르는 법령 이름이면 비슷한 이름을 알려 준다."""
    searcher = Searcher(index, _embed)

    with pytest.raises(ValueError, match="근로기준" + ".*근로기준법"):
        searcher.search("임금", laws=["근로기준"])
    assert searcher.resolve_laws(["최저임금법"]) == ["최저임금법"]


def test_list_laws_by_theme_and_article(index) -> None:
    searcher = Searcher(index, _embed)

    assert [law.law_name for law in searcher.list_laws(theme="tax")] == ["최저임금법"]
    assert len(searcher.list_laws()) == 3
    assert [h.id for h in searcher.get_article("근로_제56조")] == ["근로_제56조_제1항", "근로_제56조_제2항"]
    assert searcher.get("임대_제4조").metadata["law_name"] == "주택임대차보호법"


def test_answer_stream_events_in_order(index) -> None:
    """검색 결과 → 답 조각들 → 완료(전체 답, 토큰 수) 순서로 내보낸다."""
    stream = _FakeStream(["휴일은 ", "주 1회 [1]"])
    searcher = Searcher(index, _embed, stream=stream)

    events = list(searcher.answer_stream("휴일", k=2, laws=["근로기준법"]))

    assert isinstance(events[0], HitsEvent) and len(events[0].hits) == 2
    assert [e.text for e in events if isinstance(e, DeltaEvent)] == ["휴일은 ", "주 1회 [1]"]
    done = events[-1]
    assert isinstance(done, DoneEvent) and done.error is None
    assert done.answer == "휴일은 주 1회 [1]" and done.input_tokens == 11
    assert "[1] 근로_제55조 제목" in stream.prompts[0]


def test_answer_stream_puts_history_in_prompt(index) -> None:
    """이전 대화는 프롬프트에 넣고, 검색은 현재 질문으로만 한다."""
    stream = _FakeStream(["네"])
    searcher = Searcher(index, _embed, stream=stream)
    history = [{"role": "user", "content": "휴일이 뭐야?"}, {"role": "assistant", "content": "쉬는 날"}]

    list(searcher.answer_stream("그럼 임금은?", history=history))

    assert "[이전 대화]\n사용자: 휴일이 뭐야?\n도우미: 쉬는 날" in stream.prompts[0]


def test_answer_stream_without_llm_returns_hits_and_error(index) -> None:
    """LLM을 쓸 수 없으면 검색 결과는 보내고 이유를 담아 끝낸다."""
    searcher = Searcher(index, _embed, stream=None, llm_available=False)

    events = list(searcher.answer_stream("휴일"))

    assert isinstance(events[0], HitsEvent)
    assert isinstance(events[1], DoneEvent) and "GEMINI_API_KEY" in events[1].error


def test_answer_stream_llm_failure_keeps_partial_answer(index) -> None:
    stream = _FakeStream(["반쯤"], fail=True)
    searcher = Searcher(index, _embed, stream=stream)

    done = list(searcher.answer_stream("휴일"))[-1]

    assert done.answer == "반쯤" and "503" in done.error


def test_answer_collects_stream(index) -> None:
    """answer()는 스트림을 모은 결과를 돌려준다 (CLI·MCP용)."""
    searcher = Searcher(index, _embed, stream=_FakeStream(["가", "나"]))

    result = searcher.answer("휴일", k=1)

    assert result.answer == "가나" and len(result.hits) == 1 and result.error is None
    assert result.llm_calls == 1 and result.output_tokens == 2


def test_open_missing_index_explains(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ragkit index"):
        Searcher.open(index_path=tmp_path / "none.sqlite")


def test_hit_to_dict_flattens_metadata(index) -> None:
    hit = Searcher(index, _embed).get("임대_제4조")

    record = hit.to_dict()

    assert record["id"] == "임대_제4조" and record["law_name"] == "주택임대차보호법"
    assert record["score"] == 0.0 and record["text"] == "임대차 임차인 임대인"
