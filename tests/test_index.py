"""SQLite(sqlite-vec) 인덱스 생성·캐시·필터 검색 테스트."""

from pathlib import Path

import numpy as np
import pytest

from ragkit.retrieval import VectorIndex, build_index


class _CountingEmbed:
    """글자 기반의 결정적 가짜 임베딩. 임베딩한 텍스트를 기록한다."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.seen.extend(texts)
        vecs = np.array([[t.count("가"), t.count("나"), 0.5] for t in texts], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _doc(doc_id: str, text: str, theme: str, law_type: str = "법률", effective: str = "2020-01-01") -> dict:
    return {
        "id": doc_id,
        "parent_id": doc_id,
        "title": f"{doc_id} 제목",
        "text": text,
        "theme": theme,
        "law_name": f"{theme}법",
        "law_type": law_type,
        "category": f"{theme}법",
        "effective_date": effective,
    }


DOCS = [
    _doc("a", "가가가", "youth"),
    _doc("b", "나나나", "tax"),
    _doc("c", "가가나", "tax", law_type="시행령", effective="2030-01-01"),
]


def test_build_index_embeds_title_and_text_with_passage_prefix(tmp_path: Path) -> None:
    """문서 텍스트는 'passage: ' + 제목 + 줄바꿈 + 본문으로 임베딩한다."""
    embed = _CountingEmbed()
    build_index(DOCS, embed, tmp_path / "idx.sqlite", model_key="fake")

    assert embed.seen[0] == "passage: a 제목\n가가가"


def test_build_index_uses_model_specific_doc_format(tmp_path: Path) -> None:
    """모델 프로필의 format_doc을 넘기면 그 문자열을 그대로 임베딩한다."""
    from ragkit.embeddings.profiles import get_profile

    embed = _CountingEmbed()
    gemma = get_profile("google/embeddinggemma-300m")
    build_index(DOCS, embed, tmp_path / "idx.sqlite", model_key="g", format_doc=gemma.format_doc)

    assert embed.seen[0] == "title: a 제목 | text: 가가가"


def test_search_returns_ids_scores_and_metadata(tmp_path: Path) -> None:
    """검색 결과는 점수 내림차순이고 id, 본문, 메타데이터를 담는다."""
    embed = _CountingEmbed()
    index = build_index(DOCS, embed, tmp_path / "idx.sqlite", model_key="fake")

    hits = index.search(embed(["가가가가"])[0], k=2)

    assert [h.id for h in hits] == ["a", "c"]
    assert hits[0].score > hits[1].score
    assert hits[0].text == "가가가"
    assert hits[0].metadata["theme"] == "youth"


def test_search_filters_by_metadata(tmp_path: Path) -> None:
    """벡터 검색과 메타데이터 조건(같음, 크기 비교)을 함께 건다."""
    embed = _CountingEmbed()
    index = build_index(DOCS, embed, tmp_path / "idx.sqlite", model_key="fake")
    query = embed(["가가가가"])[0]

    assert [h.id for h in index.search(query, k=3, where={"theme": "tax"})] == ["c", "b"]
    current = index.search(query, k=3, where={"effective_date": ("<=", "2026-09-29")})
    assert [h.id for h in current] == ["a", "b"]


def test_build_index_reuses_existing_file(tmp_path: Path) -> None:
    """같은 모델 키와 같은 코퍼스면 파일을 그대로 쓰고 다시 임베딩하지 않는다."""
    embed = _CountingEmbed()
    build_index(DOCS, embed, tmp_path / "idx.sqlite", model_key="fake")
    build_index(DOCS, embed, tmp_path / "idx.sqlite", model_key="fake")

    assert len(embed.seen) == 3


def test_build_index_rebuilds_when_corpus_or_model_changes(tmp_path: Path) -> None:
    """코퍼스나 모델 키가 바뀌면 파일을 새로 만든다."""
    embed = _CountingEmbed()
    path = tmp_path / "idx.sqlite"
    build_index(DOCS, embed, path, model_key="fake")
    build_index([*DOCS[:2], _doc("c", "나가", "tax")], embed, path, model_key="fake")
    index = build_index(DOCS, embed, path, model_key="other")

    assert len(embed.seen) == 9
    assert index.model_key == "other"


def test_build_index_replaces_invalid_existing_file(tmp_path: Path) -> None:
    """인덱스가 아닌 파일(빈 파일, 옛 형식)이 있으면 새로 만든다."""
    import sqlite3

    path = tmp_path / "idx.sqlite"
    sqlite3.connect(str(path)).close()  # 빈 DB 파일
    embed = _CountingEmbed()

    index = build_index(DOCS, embed, path, model_key="fake")

    assert len(index) == 3


def test_build_index_rejects_duplicate_ids_before_embedding(tmp_path: Path) -> None:
    """id가 겹치면 임베딩을 시작하기 전에 어떤 id인지 알려 주고 멈춘다."""
    embed = _CountingEmbed()

    with pytest.raises(ValueError, match="중복 id 1개.*a"):
        build_index([*DOCS, _doc("a", "다른 본문", "youth")], embed, tmp_path / "idx.sqlite", model_key="fake")

    assert embed.seen == []


def test_open_existing_index_without_embedding(tmp_path: Path) -> None:
    """앱은 만들어 둔 파일을 열어 검색만 한다."""
    embed = _CountingEmbed()
    build_index(DOCS, embed, tmp_path / "idx.sqlite", model_key="fake")

    index = VectorIndex.open(tmp_path / "idx.sqlite")

    assert len(index) == 3
    assert index.search(embed(["나나나"])[0], k=1)[0].id == "b"


def test_open_missing_index_explains_how_to_build(tmp_path: Path) -> None:
    """인덱스 파일이 없으면 만드는 방법을 안내한다."""
    with pytest.raises(FileNotFoundError, match="ragkit index"):
        VectorIndex.open(tmp_path / "none.sqlite")
