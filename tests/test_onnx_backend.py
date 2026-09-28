"""ONNX 임베딩 백엔드 테스트."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ragkit.config import get_settings

LOCAL_MODEL = get_settings().models_dir / "multilingual-e5-small"


def test_numpy_mean_pool_ignores_padding() -> None:
    """패딩 토큰 값은 평균에 포함되지 않는다 (numpy 버전)."""
    from ragkit.embeddings.onnx_backend import mean_pool

    hidden = np.array([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = np.array([[1, 1, 0]])

    np.testing.assert_allclose(mean_pool(hidden, mask), [[2.0, 2.0]])


def test_import_embeddings_does_not_import_torch() -> None:
    """core 배포(torch 없음)에서도 ragkit.embeddings를 import할 수 있어야 한다."""
    code = "import sys, ragkit.embeddings, ragkit.models; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_onnx_backend_requires_exported_model(tmp_path: Path) -> None:
    """ONNX 파일이 없으면 export-onnx 실행을 안내한다."""
    from ragkit.embeddings import create_embedding_fn

    with pytest.raises(FileNotFoundError, match="export-onnx"):
        create_embedding_fn(str(tmp_path), backend="onnx")


@pytest.mark.skipif(not LOCAL_MODEL.exists(), reason="로컬 모델 없음")
def test_onnx_matches_torch(tmp_path: Path) -> None:
    """변환한 ONNX 모델의 임베딩이 torch 임베딩과 거의 같다 (코사인 ≥ 0.999)."""
    from ragkit.embeddings import create_embedding_fn, format_queries
    from ragkit.models.onnx_export import export_onnx

    onnx_path = export_onnx(LOCAL_MODEL, tmp_path)
    assert onnx_path == tmp_path / "onnx" / "model.onnx"

    texts = format_queries(
        ["편의점 알바 주휴수당 받을 수 있나요?", "전세 계약 끝나면 보증금은 언제 돌려받아요?", "최저임금"]
    )
    torch_emb = create_embedding_fn(str(LOCAL_MODEL), backend="torch")(texts)
    onnx_emb = create_embedding_fn(str(tmp_path), backend="onnx")(texts)

    assert onnx_emb.shape == torch_emb.shape
    np.testing.assert_allclose(np.linalg.norm(onnx_emb, axis=1), 1.0, atol=1e-5)
    cosines = (onnx_emb * torch_emb).sum(axis=1)
    assert cosines.min() >= 0.999

    # 패딩이 없는 입력(한 문장)도 같은 결과여야 한다 (변환 시 패딩 분기가 고정되지 않았는지 확인)
    single = create_embedding_fn(str(tmp_path), backend="onnx")(texts[:1])
    assert float(single[0] @ torch_emb[0]) >= 0.999
