"""ONNX 임베딩 백엔드 (배포 기본값, torch 불필요).

모델 폴더 구성:
    <model_dir>/tokenizer.json
    <model_dir>/onnx/model.onnx   ← `ragkit export-onnx <model_dir>`로 생성
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MAX_LENGTH = 512


def mean_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """패딩 토큰을 제외하고 토큰 임베딩의 평균을 구한다.

    Args:
        hidden_states: (batch, seq_len, dim) 형태의 마지막 은닉 상태.
        attention_mask: (batch, seq_len) 형태의 어텐션 마스크 (패딩=0).

    Returns:
        (batch, dim) 형태의 문장 임베딩.
    """
    mask = attention_mask[..., None].astype(hidden_states.dtype)
    return (hidden_states * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)


def onnx_model_path(model_dir: Path) -> Path:
    """모델 폴더 안의 ONNX 파일 경로를 반환한다."""
    return model_dir / "onnx" / "model.onnx"


def create_onnx_embedding_fn(model_dir: Path):
    """ONNX 모델을 로드하고 임베딩 생성 함수를 반환한다.

    Args:
        model_dir: tokenizer.json과 onnx/model.onnx가 있는 모델 폴더.

    Returns:
        texts를 받아 L2 정규화된 (N, dim) np.ndarray를 반환하는 함수.

    Raises:
        FileNotFoundError: ONNX 파일이나 tokenizer.json이 없을 때.
    """
    onnx_path = onnx_model_path(model_dir)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX 모델이 없습니다: {onnx_path}\n"
            f"먼저 변환하세요: uv run ragkit export-onnx {model_dir}"
        )
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json이 없습니다: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}

    def _embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            encoded = tokenizer.encode_batch(texts[i : i + batch_size])
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)
            hidden = session.run(None, feeds)[0]
            pooled = mean_pool(hidden, attention_mask)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            embeddings.append(pooled.astype(np.float32))
        return np.concatenate(embeddings, axis=0)

    return _embed
