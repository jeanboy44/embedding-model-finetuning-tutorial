"""임베딩 속도 옵션 테스트: 길이 정렬 배치, 장치 자동 선택."""

import numpy as np
import pytest

from ragkit.config import get_settings
from ragkit.embeddings.batching import embed_in_batches


def _identity_embed(batches: list[list[str]]):
    def embed_batch(texts: list[str]) -> np.ndarray:
        batches.append(list(texts))
        return np.array([[len(t), ord(t[0])] for t in texts], dtype=np.float32)

    return embed_batch


TEXTS = ["aaaa", "b", "ccccccc", "dd", "eeeee"]


def test_sorted_batches_group_similar_lengths() -> None:
    """길이순으로 묶어 배치 안의 패딩을 줄인다."""
    batches: list[list[str]] = []

    embed_in_batches(TEXTS, _identity_embed(batches), batch_size=2, sort_by_length=True)

    assert batches == [["b", "dd"], ["aaaa", "eeeee"], ["ccccccc"]]


def test_sorted_batches_return_original_order() -> None:
    """정렬해서 계산해도 결과는 입력 순서대로 돌려준다."""
    out = embed_in_batches(TEXTS, _identity_embed([]), batch_size=2, sort_by_length=True)

    np.testing.assert_array_equal(out[:, 0], [len(t) for t in TEXTS])
    np.testing.assert_array_equal(out[:, 1], [ord(t[0]) for t in TEXTS])


def test_unsorted_batches_keep_input_order() -> None:
    """정렬을 끄면 입력 순서대로 배치를 만든다 (비교 실험용)."""
    batches: list[list[str]] = []

    embed_in_batches(TEXTS, _identity_embed(batches), batch_size=2, sort_by_length=False)

    assert batches == [["aaaa", "b"], ["ccccccc", "dd"], ["eeeee"]]


LOCAL_MODEL = get_settings().models_dir / "multilingual-e5-small"
MIXED = ["짧은 문장", "조금 더 긴 문장입니다. " * 3, "가", "아주 긴 조문 본문 " * 40, "중간 길이의 문장이에요"]


@pytest.mark.skipif(not (LOCAL_MODEL / "onnx" / "model.onnx").exists(), reason="로컬 ONNX 모델 없음")
@pytest.mark.parametrize("backend", ["onnx", "torch"])
def test_sorting_does_not_change_embeddings(backend: str) -> None:
    """길이 정렬은 속도만 바꾸고 결과(순서 포함)는 그대로다."""
    from ragkit.embeddings import create_embedding_fn

    sorted_fn = create_embedding_fn(str(LOCAL_MODEL), backend=backend, sort_by_length=True)
    plain_fn = create_embedding_fn(str(LOCAL_MODEL), backend=backend, sort_by_length=False)

    a, b = sorted_fn(MIXED, batch_size=2), plain_fn(MIXED, batch_size=2)

    assert ((a * b).sum(axis=1)).min() >= 0.999


def test_resolve_device_prefers_cuda_then_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto는 cuda → mps → cpu 순으로 고른다. 직접 지정하면 그대로 쓴다."""
    torch = pytest.importorskip("torch")
    from ragkit.embeddings.torch_backend import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"
    assert resolve_device(None) == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
