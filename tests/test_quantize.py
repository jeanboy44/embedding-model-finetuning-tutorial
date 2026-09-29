"""ONNX 동적 INT8 양자화 테스트 (로컬 ONNX 모델 사용)."""

from pathlib import Path

import pytest

from ragkit.config import get_settings

LOCAL = get_settings().models_dir / "multilingual-e5-small"


@pytest.mark.skipif(not (LOCAL / "onnx" / "model.onnx").exists(), reason="로컬 ONNX 모델 없음")
def test_quantize_makes_smaller_model_folder_with_similar_embeddings(tmp_path: Path) -> None:
    """INT8 모델은 별도 폴더(토크나이저 포함)에 생기고, 크기는 줄고, 임베딩은 원본과 비슷하다."""
    from ragkit.embeddings import create_embedding_fn
    from ragkit.models.onnx_export import quantize_onnx

    out = quantize_onnx(LOCAL, tmp_path / "e5-int8")

    assert out == tmp_path / "e5-int8" / "onnx" / "model.onnx"
    assert (tmp_path / "e5-int8" / "tokenizer.json").exists()
    assert out.stat().st_size < (LOCAL / "onnx" / "model.onnx").stat().st_size * 0.5

    texts = ["query: 편의점 알바 주휴수당", "passage: 근로기준법 제55조 (휴일)\n유급휴일을 보장하여야 한다."]
    fp32 = create_embedding_fn(str(LOCAL), backend="onnx")(texts)
    int8 = create_embedding_fn(str(tmp_path / "e5-int8"), backend="onnx")(texts)
    assert (fp32 * int8).sum(axis=1).min() >= 0.995  # 채널별 양자화: 텐서 단위(약 0.98)보다 원본에 가깝다


def test_quantize_requires_exported_onnx(tmp_path: Path) -> None:
    """ONNX 파일이 없으면 export-onnx를 먼저 하라고 안내한다."""
    from ragkit.models.onnx_export import quantize_onnx

    with pytest.raises(FileNotFoundError, match="export-onnx"):
        quantize_onnx(tmp_path, tmp_path / "out")
