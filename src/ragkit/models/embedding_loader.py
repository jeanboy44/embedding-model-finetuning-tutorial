"""임베딩 모델 로더 (PyTorch 경로, extra `[torch]` 필요).

torch/transformers는 함수 안에서 import한다. core 배포(ONNX)에서도
`import ragkit.models`가 깨지지 않게 하기 위함이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ragkit.config import get_settings

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase

TORCH_EXTRA_HINT = "PyTorch 백엔드에는 extra가 필요합니다: uv sync --extra torch"


def resolve_model_source(model_name: str) -> str:
    """로컬 models/ 폴더에 받아둔 모델이 있으면 그 경로를, 없으면 원래 이름을 반환한다.

    scripts/download_model_hf.py 등으로 받아둔 모델을 오프라인에서도 쓰기 위함이다.

    Args:
        model_name: HuggingFace 모델 이름 (예: "intfloat/multilingual-e5-small")
            또는 로컬 경로.

    Returns:
        from_pretrained에 넘길 로컬 경로 또는 모델 이름.
    """
    if (Path(model_name) / "config.json").exists():
        return model_name
    local_dir = get_settings().models_dir / model_name.split("/")[-1]
    if (local_dir / "config.json").exists():
        return str(local_dir)
    return model_name


def _import_torch():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(TORCH_EXTRA_HINT) from e
    return torch, AutoModel, AutoTokenizer


def resolve_device(device: str | None) -> str:
    """장치 이름을 정한다. None/"auto"면 cuda → mps(Apple GPU) → cpu 순으로 고른다."""
    if device not in (None, "auto"):
        return device
    try:
        import torch
    except ImportError as e:
        raise ImportError(TORCH_EXTRA_HINT) from e
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedding_model(
    model_name: str,
    checkpoint_path: Path | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """로컬 models/, HuggingFace 또는 체크포인트에서 임베딩 모델을 로드한다.

    Args:
        model_name: HuggingFace 모델 이름 (예: "intfloat/multilingual-e5-small").
        checkpoint_path: 파인튜닝된 체크포인트. 폴더(HF/sentence-transformers 형식)면
            그 폴더에서 모델을 읽고, 파일이면 state_dict로 덮어쓴다. None이면 기본 모델.
        device: 디바이스 문자열 (예: "cuda", "mps", "cpu"). None/"auto"면 자동 감지.

    Returns:
        평가 모드로 설정된 모델.
    """
    torch, AutoModel, _ = _import_torch()
    device = resolve_device(device)

    if checkpoint_path and Path(checkpoint_path).is_dir():
        model = AutoModel.from_pretrained(str(checkpoint_path))
    else:
        model = AutoModel.from_pretrained(resolve_model_source(model_name))
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()
    return model


def load_tokenizer(
    model_name: str, checkpoint_path: Path | None = None
) -> PreTrainedTokenizerBase:
    """임베딩 모델용 토크나이저를 로드한다.

    Args:
        model_name: HuggingFace 모델 이름 또는 로컬 경로.
        checkpoint_path: 체크포인트 폴더에 토크나이저가 있으면 그것을 쓴다.

    Returns:
        로드된 토크나이저.
    """
    _, _, AutoTokenizer = _import_torch()
    if checkpoint_path and (Path(checkpoint_path) / "tokenizer.json").exists():
        return AutoTokenizer.from_pretrained(str(checkpoint_path))
    return AutoTokenizer.from_pretrained(resolve_model_source(model_name))
