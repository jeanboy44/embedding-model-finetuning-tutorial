"""인덱스 생성·캐시와 id 포함 검색 테스트."""

from pathlib import Path

import numpy as np

from ragkit.retrieval import build_index, search


class _CountingEmbed:
    """글자 기반의 결정적 가짜 임베딩. 호출된 텍스트 수를 센다."""

    def __init__(self) -> None:
        self.embedded = 0

    def __call__(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.embedded += len(texts)
        vecs = np.array([[t.count("가"), t.count("나"), 1.0] for t in texts], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


IDS = ["a", "b", "c"]
TEXTS = ["가가가", "나나나", "가나"]


def test_build_index_adds_passage_prefix_and_ids() -> None:
    """문서에는 passage 앞 문구를 붙여 임베딩하고, 검색 결과는 id와 원문을 돌려준다."""
    seen: list[str] = []

    def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
        seen.extend(texts)
        return _CountingEmbed()(texts)

    store = build_index(IDS, TEXTS, embed)

    assert seen[0] == "passage: 가가가"
    hits = search(store, embed(["가가가가"]), k=2)
    assert [h.id for h in hits] == ["a", "c"]
    assert hits[0].text == "가가가"
    assert hits[0].score > hits[1].score


def test_build_index_reuses_cache(tmp_path: Path) -> None:
    """같은 모델 키와 같은 코퍼스면 캐시를 읽고 다시 임베딩하지 않는다."""
    embed = _CountingEmbed()
    build_index(IDS, TEXTS, embed, cache_dir=tmp_path, cache_key="e5-small-onnx")
    first = embed.embedded

    store = build_index(IDS, TEXTS, embed, cache_dir=tmp_path, cache_key="e5-small-onnx")

    assert first == 3
    assert embed.embedded == 3
    assert [h.id for h in search(store, embed(["나나"]), k=1)] == ["b"]


def test_build_index_cache_depends_on_corpus_and_model(tmp_path: Path) -> None:
    """코퍼스나 모델 키가 바뀌면 캐시를 쓰지 않고 새로 만든다."""
    embed = _CountingEmbed()
    build_index(IDS, TEXTS, embed, cache_dir=tmp_path, cache_key="m1")
    build_index(IDS, [*TEXTS[:2], "나가"], embed, cache_dir=tmp_path, cache_key="m1")
    build_index(IDS, TEXTS, embed, cache_dir=tmp_path, cache_key="m2")

    assert embed.embedded == 9
