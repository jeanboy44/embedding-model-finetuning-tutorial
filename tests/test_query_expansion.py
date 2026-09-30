"""LLM 쿼리 확장 테스트 (가짜 LLM)."""

import json

import pytest

from ragkit.models import Generation
from ragkit.rag.query_expansion import (
    ExpansionUnavailable,
    expand_queries,
    expansion_prompt,
    expansion_summary,
    load_cache,
    search_text,
)


class _FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> Generation:
        self.prompts.append(prompt)
        return Generation(text="  해고 예고 근로기준법\n", input_tokens=40, output_tokens=6)


def test_prompt_asks_for_legal_terms_only() -> None:
    prompt = expansion_prompt("알바 잘렸는데 돈 받을 수 있어?")

    assert "알바 잘렸는데 돈 받을 수 있어?" in prompt
    assert "법률 용어" in prompt
    assert "검색어만" in prompt


def test_search_text_keeps_original_question() -> None:
    """확장어만 쓰면 원래 질문의 뜻이 빠질 수 있어 원문 뒤에 붙인다."""
    assert search_text("알바 잘렸어", "해고 예고") == "알바 잘렸어 해고 예고"


def test_expand_queries_calls_llm_once_per_new_query_and_caches(tmp_path) -> None:
    cache = tmp_path / "cache.jsonl"
    llm = _FakeLLM()

    first = expand_queries(["질문1", "질문2", "질문1"], cache, generate=llm, llm="fake-llm")

    assert len(llm.prompts) == 2  # 중복 질문은 한 번만
    assert first["질문1"].expansion == "해고 예고 근로기준법"  # 앞뒤 공백·줄바꿈 제거
    assert first["질문1"].llm == "fake-llm"
    assert first["질문1"].input_tokens == 40

    again = expand_queries(["질문1", "질문2"], cache, generate=llm, llm="fake-llm")

    assert len(llm.prompts) == 2  # 캐시에 있으면 부르지 않는다
    assert again == first
    assert set(load_cache(cache)) == {"질문1", "질문2"}


def test_expand_queries_resumes_after_failure(tmp_path) -> None:
    """한도 초과 등으로 중간에 멈춰도 그때까지 받은 확장은 캐시에 남는다."""
    cache = tmp_path / "cache.jsonl"
    calls = []

    def flaky(prompt: str) -> Generation:
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("quota")
        return Generation(text="확장", input_tokens=1, output_tokens=1)

    with pytest.raises(RuntimeError):
        expand_queries(["a", "b", "c"], cache, generate=flaky, llm="fake")

    assert set(load_cache(cache)) == {"a"}


def test_expand_queries_waits_between_calls(tmp_path) -> None:
    """rpm 한도가 있는 무료 등급을 위해 호출 사이에 min_interval_s를 지킨다."""
    waits: list[float] = []
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    expand_queries(
        ["a", "b", "c"],
        tmp_path / "cache.jsonl",
        generate=_FakeLLM(),
        llm="fake",
        min_interval_s=4.0,
        sleep=sleep,
        clock=clock,
    )

    assert waits == [4.0, 4.0]  # 첫 호출은 기다리지 않는다


def test_expand_queries_without_api_key_needs_full_cache(tmp_path, monkeypatch) -> None:
    """키가 없어도 캐시에 모두 있으면 된다(강의에서 배포한 캐시). 모자라면 이유를 알려 준다."""
    from ragkit.config import get_settings

    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    cache = tmp_path / "cache.jsonl"
    expand_queries(["a"], cache, generate=_FakeLLM(), llm="fake")

    assert expand_queries(["a"], cache, llm="fake")["a"].expansion

    with pytest.raises(ExpansionUnavailable, match="1개"):
        expand_queries(["a", "b"], cache, llm="fake")
    get_settings.cache_clear()


def test_load_cache_ignores_other_llm_and_missing_file(tmp_path) -> None:
    cache = tmp_path / "cache.jsonl"
    assert load_cache(cache) == {}
    row = {"query": "a", "expansion": "x", "llm": "other", "input_tokens": 1, "output_tokens": 1, "latency_s": 0.1}
    cache.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    assert load_cache(cache, llm="fake") == {}
    assert set(load_cache(cache)) == {"a"}


def test_expansion_summary_reports_cost_per_query(tmp_path) -> None:
    expansions = expand_queries(["a", "b"], tmp_path / "c.jsonl", generate=_FakeLLM(), llm="fake")

    summary = expansion_summary(list(expansions.values()))

    assert summary["llm"] == "fake"
    assert summary["llm_calls_per_query"] == 1
    assert summary["input_tokens_mean"] == 40
    assert summary["output_tokens_mean"] == 6
    assert summary["latency_s_mean"] >= 0
