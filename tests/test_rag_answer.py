"""RAG 답변 / 순수 LLM 답변 테스트 (가짜 임베딩·가짜 LLM)."""

from pathlib import Path

import numpy as np

from ragkit.models import Generation
from ragkit.rag import answer_with_rag, answer_without_retrieval, build_prompt
from ragkit.retrieval import SearchHit, build_index


def _embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    vecs = np.array([[t.count("휴"), t.count("임"), 0.5] for t in texts], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class _FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> Generation:
        self.prompts.append(prompt)
        return Generation(text="답변입니다", input_tokens=len(prompt), output_tokens=3)


DOCS = [
    {"id": "근로기준법_제55조", "title": "근로기준법 제55조 (휴일)", "text": "유급휴일 휴일 휴일", "theme": "youth"},
    {"id": "주택임대차보호법_제4조", "title": "주택임대차보호법 제4조", "text": "임대차 임차인 임대인", "theme": "youth"},
]


def test_build_prompt_numbers_sources_and_asks_to_ground() -> None:
    """프롬프트에 조문을 번호 붙여 넣고, 조문에 근거해 답하라고 지시한다."""
    hits = [SearchHit(id="a", text="본문A", score=0.9, metadata={"title": "제목A"})]

    prompt = build_prompt("질문?", hits)

    assert "[1] 제목A\n본문A" in prompt
    assert "질문?" in prompt
    assert "조문" in prompt and "근거" in prompt


def test_answer_with_rag_retrieves_then_generates(tmp_path: Path) -> None:
    """검색한 조문으로 프롬프트를 만들어 LLM을 1회 호출하고 토큰 수를 돌려준다."""
    index = build_index(DOCS, _embed, tmp_path / "idx.sqlite", model_key="fake")
    llm = _FakeLLM()

    result = answer_with_rag("휴일 수당", index, _embed, generate=llm, k=1)

    assert [h.id for h in result.hits] == ["근로기준법_제55조"]
    assert "근로기준법 제55조 (휴일)" in llm.prompts[0]
    assert result.answer == "답변입니다"
    assert result.llm_calls == 1
    assert result.input_tokens == len(llm.prompts[0])
    assert result.latency_s >= 0


def test_answer_without_retrieval_sends_question_only() -> None:
    """순수 LLM은 검색 없이 질문만 보낸다."""
    llm = _FakeLLM()

    result = answer_without_retrieval("주휴수당 받을 수 있나요?", generate=llm)

    assert result.hits == []
    assert llm.prompts == ["주휴수당 받을 수 있나요?"]
    assert result.llm_calls == 1
