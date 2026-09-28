"""Gemini 클라이언트의 토큰 사용량·토큰 수 테스트 (실제 API 대신 가짜 클라이언트)."""

from types import SimpleNamespace

import pytest

from ragkit.models import gemini_client


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents})
        usage = SimpleNamespace(prompt_token_count=12, candidates_token_count=5)
        return SimpleNamespace(text="답변", usage_metadata=usage)

    def count_tokens(self, *, model, contents):
        return SimpleNamespace(total_tokens=len(contents))


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeModels:
    models = _FakeModels()
    monkeypatch.setattr(gemini_client, "_get_client", lambda api_key=None: SimpleNamespace(models=models))
    return models


def test_generate_with_usage_returns_text_and_token_counts(fake_client: _FakeModels) -> None:
    """답변과 함께 입력·출력 토큰 수를 돌려준다."""
    result = gemini_client.generate_with_usage("질문", model_name="m")

    assert result.text == "답변"
    assert result.input_tokens == 12
    assert result.output_tokens == 5
    assert fake_client.calls == [{"model": "m", "contents": "질문"}]


def test_count_tokens_uses_api(fake_client: _FakeModels) -> None:
    """LLM 토크나이저 기준 토큰 수를 센다."""
    assert gemini_client.count_tokens("가나다라", model_name="m") == 4
