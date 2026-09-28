"""HuggingFace Hub에서 임베딩 모델을 내려받아 로컬 models/ 폴더에 저장한다.

사용법:
    uv run python scripts/download_model_hf.py
    uv run python scripts/download_model_hf.py --force

저장된 모델은 src.models.load_embedding_model이 자동으로 찾아 사용한다.
"""

from pathlib import Path

from cyclopts import run
from huggingface_hub import snapshot_download

from src.config import get_settings

# 추론에 필요한 파일만 받는다 (pytorch_model.bin, onnx/, openvino/ 등 중복 가중치 제외).
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",
]


def download_from_hf(
    repo_id: str,
    output_dir: Path,
    *,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """HuggingFace Hub에서 모델을 내려받는다.

    Args:
        repo_id: HuggingFace 모델 저장소 이름.
        output_dir: 모델을 저장할 디렉토리.
        revision: 브랜치, 태그 또는 커밋 해시. None이면 main.
        force: True면 이미 받은 모델이 있어도 다시 받는다.

    Returns:
        모델이 저장된 디렉토리 경로.
    """
    if (output_dir / "config.json").exists() and not force:
        print(f"이미 존재합니다: {output_dir} (다시 받으려면 --force)")
        return output_dir

    print(f"다운로드 중: {repo_id} → {output_dir}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=output_dir,
        allow_patterns=MODEL_FILES,
    )
    print(f"완료: {output_dir}")
    return output_dir


def main(
    repo_id: str | None = None,
    output_dir: Path | None = None,
    revision: str | None = None,
    force: bool = False,
) -> None:
    """HuggingFace Hub에서 임베딩 모델을 내려받는다.

    Args:
        repo_id: HuggingFace 모델 저장소 이름. 기본값은 설정의 embedding_model_name.
        output_dir: 저장 디렉토리. 기본값은 models/<모델 이름>.
        revision: 브랜치, 태그 또는 커밋 해시.
        force: 이미 받은 모델이 있어도 다시 받는다.
    """
    settings = get_settings()
    repo_id = repo_id or settings.embedding_model_name
    output_dir = output_dir or settings.models_dir / repo_id.split("/")[-1]
    download_from_hf(repo_id, output_dir, revision=revision, force=force)


if __name__ == "__main__":
    run(main)
