"""앱 테스트 공용: 작은 가짜 법령 인덱스 + 가짜 임베딩 + 가짜 LLM 스트림으로 만든 Searcher."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from ragkit.models import Generation
from ragkit.retrieval import build_index
from ragkit.service import Searcher


def fake_embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    vecs = np.array([[t.count("휴"), t.count("임"), 0.5] for t in texts], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _doc(doc_id: str, law: str, text: str, parent: str | None = None, theme: str = "youth") -> dict:
    return {"id": doc_id, "parent_id": parent or doc_id, "title": f"{law} {doc_id}", "text": text,
            "law_name": law, "law_type": "법률", "theme": theme, "article_no": doc_id,
            "source_url": f"https://www.law.go.kr/법령/{law}"}


DOCS = [
    _doc("제55조", "근로기준법", "휴일 휴일 휴일"),
    _doc("제56조_제1항", "근로기준법", "휴일 근로 임금", parent="제56조"),
    _doc("제56조_제2항", "근로기준법", "연장 근로", parent="제56조"),
    _doc("제4조", "주택임대차보호법", "임대차 임차인 임대인"),
    _doc("제6조", "최저임금법", "임금 임금 휴", theme="tax"),
]


class FakeStream:
    """정해 둔 조각을 내보내는 가짜 LLM. 받은 프롬프트를 기록한다."""

    def __init__(self, pieces: list[str] | None = None) -> None:
        self.pieces = pieces or ["휴일에는 ", "가산 임금을 받습니다 [1]"]
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> Iterator[str | Generation]:
        self.prompts.append(prompt)
        yield from self.pieces
        yield Generation(text="".join(self.pieces), input_tokens=20, output_tokens=5)


@pytest.fixture
def fake_stream() -> FakeStream:
    return FakeStream()


@pytest.fixture
def searcher(tmp_path: Path, fake_stream: FakeStream) -> Searcher:
    index = build_index(DOCS, fake_embed, tmp_path / "idx.sqlite", model_key="fake")
    return Searcher(index, fake_embed, stream=fake_stream)


@pytest.fixture
def searcher_without_llm(tmp_path: Path) -> Searcher:
    index = build_index(DOCS, fake_embed, tmp_path / "idx2.sqlite", model_key="fake")
    return Searcher(index, fake_embed, llm_available=False)
