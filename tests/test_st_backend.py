"""sentence-transformers 백엔드와 모델별 백엔드 선택 테스트."""

import pytest

from ragkit.config import get_settings
from ragkit.embeddings import choose_backend
from ragkit.embeddings.profiles import get_profile

LOCAL_E5 = get_settings().models_dir / "multilingual-e5-small"


def test_choose_backend_uses_default_when_supported() -> None:
    """설정 기본값(onnx)을 지원하는 모델은 그대로 쓴다."""
    assert choose_backend(get_profile("intfloat/multilingual-e5-small"), None) == "onnx"


def test_choose_backend_falls_back_to_supported_one() -> None:
    """기본값을 지원하지 않는 모델(EmbeddingGemma)은 지원하는 백엔드(st)로 바꾼다."""
    assert choose_backend(get_profile("google/embeddinggemma-300m"), None) == "st"


def test_choose_backend_rejects_explicit_unsupported() -> None:
    """직접 지정한 백엔드를 모델이 지원하지 않으면 이유를 알려 준다."""
    with pytest.raises(ValueError, match="embeddinggemma.*st"):
        choose_backend(get_profile("google/embeddinggemma-300m"), "torch")


def test_st_backend_disables_model_default_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """모델의 기본 프롬프트(jina: 'Document: ')를 끈다. 형식은 프로필이 이미 붙였다(이중 접두어 방지)."""
    import numpy as np
    import sentence_transformers

    from ragkit.embeddings.st_backend import create_st_embedding_fn

    calls: list[dict] = []

    class FakeST:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, texts, **kwargs):
            calls.append(kwargs)
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    embed = create_st_embedding_fn("jinaai/jina-embeddings-v5-text-small",
                                   get_profile("jinaai/jina-embeddings-v5-text-small"), device="cpu")

    embed(["Query: 주휴수당"])

    assert calls[0]["prompt"] == ""
    assert calls[0]["task"] == "retrieval"


@pytest.mark.skipif(not LOCAL_E5.exists(), reason="로컬 모델 없음")
def test_st_backend_matches_torch_for_e5() -> None:
    """같은 e5 모델이면 st 백엔드와 torch 백엔드 결과가 같다."""
    from ragkit.embeddings import create_embedding_fn

    texts = ["query: 주휴수당", "passage: 근로기준법 제55조 (휴일)\n유급휴일을 보장하여야 한다."]
    st = create_embedding_fn(str(LOCAL_E5), backend="st")(texts)
    torch_ = create_embedding_fn(str(LOCAL_E5), backend="torch")(texts)

    assert st.shape == (2, 384)
    assert (st * torch_).sum(axis=1).min() >= 0.999
