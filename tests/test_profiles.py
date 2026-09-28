"""모델 프로필(모델별 입력 형식·백엔드·라이선스)과 모델 키 규칙 테스트."""

import os
from pathlib import Path

import pytest

from ragkit.embeddings.profiles import get_profile
from ragkit.retrieval import default_index_path, model_key

DOC = {"id": "x", "title": "근로기준법 제55조 (휴일)", "text": "유급휴일을 보장하여야 한다."}


def test_e5_profile_formats() -> None:
    """e5는 query:/passage: 앞 문구, 문서는 제목+줄바꿈+본문."""
    p = get_profile("intfloat/multilingual-e5-small")

    assert p.format_query("주휴수당") == "query: 주휴수당"
    assert p.format_doc(DOC) == "passage: 근로기준법 제55조 (휴일)\n유급휴일을 보장하여야 한다."
    assert "onnx" in p.backends and p.deployable


def test_embeddinggemma_profile_uses_title_field() -> None:
    """EmbeddingGemma는 검색용 쿼리 형식과 title/text 문서 형식을 쓴다."""
    p = get_profile("google/embeddinggemma-300m")

    assert p.format_query("주휴수당") == "task: search result | query: 주휴수당"
    assert p.format_doc(DOC) == "title: 근로기준법 제55조 (휴일) | text: 유급휴일을 보장하여야 한다."
    assert p.backends == ("st",)
    assert p.dtype == "float32"


def test_unknown_model_falls_back_to_e5_style(tmp_path: Path) -> None:
    """파인튜닝 폴더처럼 모르는 이름은 e5 형식(Settings의 앞 문구)을 쓴다."""
    p = get_profile(str(tmp_path / "exp_002_finetuned"))

    assert p.format_query("q") == "query: q"
    assert p.backends == ("onnx", "torch", "st")


def test_unknown_hub_model_is_rejected() -> None:
    """프로필이 없는 허브 모델은 형식을 모르므로 조용히 e5 형식으로 처리하지 않고 멈춘다."""
    with pytest.raises(ValueError, match="jina.*프로필이 없습니다"):
        get_profile("jinaai/jina-embeddings-v5-text-small")


def test_model_key_for_hub_name_and_checkpoint(tmp_path: Path) -> None:
    """허브 모델은 이름 끝부분, 체크포인트 폴더는 폴더 이름 + 가중치 수정 시각."""
    ckpt = tmp_path / "exp_002_finetuned"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"w")
    os.utime(ckpt / "model.safetensors", (1_790_000_000, 1_790_000_000))

    assert model_key("intfloat/multilingual-e5-small") == "multilingual-e5-small"
    assert model_key("intfloat/multilingual-e5-small", ckpt).startswith("exp_002_finetuned-")
    assert model_key("intfloat/multilingual-e5-small", ckpt) == model_key("x", ckpt)


def test_default_index_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """인덱스 파일은 data/processed/index/<모델 키>.sqlite."""
    from ragkit.config import get_settings

    monkeypatch.setenv("RAGKIT_PROJECT_ROOT", str(tmp_path))
    get_settings.cache_clear()
    try:
        assert default_index_path("m") == tmp_path / "data" / "processed" / "index" / "m.sqlite"
    finally:
        monkeypatch.delenv("RAGKIT_PROJECT_ROOT")
        get_settings.cache_clear()
