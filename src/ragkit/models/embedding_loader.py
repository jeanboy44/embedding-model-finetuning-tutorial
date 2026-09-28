"""임베딩 모델 로더."""

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from ragkit.config import get_settings


def resolve_model_source(model_name: str) -> str:
    """로컬 models/ 폴더에 받아둔 모델이 있으면 그 경로를, 없으면 원래 이름을 반환한다.

    scripts/download_model_hf.py 등으로 받아둔 모델을 오프라인에서도 쓰기 위함이다.

    Args:
        model_name: HuggingFace 모델 이름 (예: "intfloat/multilingual-e5-small")
            또는 로컬 경로.

    Returns:
        from_pretrained에 넘길 로컬 경로 또는 모델 이름.
    """
    local_dir = get_settings().models_dir / model_name.split("/")[-1]
    if (local_dir / "config.json").exists():
        return str(local_dir)
    return model_name


def load_embedding_model(
    model_name: str,
    checkpoint_path: Path | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """로컬 models/, HuggingFace 또는 체크포인트에서 임베딩 모델을 로드한다.

    Args:
        model_name: HuggingFace 모델 이름 (예: "intfloat/multilingual-e5-small").
        checkpoint_path: 파인튜닝된 체크포인트 경로. None이면 기본 모델 사용.
        device: 디바이스 문자열 (예: "cuda", "cpu"). None이면 자동 감지.

    Returns:
        평가 모드로 설정된 모델.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(resolve_model_source(model_name))

    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()
    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """임베딩 모델용 토크나이저를 로드한다.

    Args:
        model_name: HuggingFace 모델 이름 또는 로컬 경로.

    Returns:
        로드된 토크나이저.
    """
    return AutoTokenizer.from_pretrained(resolve_model_source(model_name))
