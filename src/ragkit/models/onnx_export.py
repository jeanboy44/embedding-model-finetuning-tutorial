"""임베딩 모델 → ONNX 변환 (extra `[train]` 필요).

흐름: 학습(torch) → `export_onnx` → 배포(onnxruntime, torch 불필요).
"""

import shutil
from pathlib import Path

from ragkit.embeddings.onnx_backend import onnx_model_path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)


def export_onnx(model_dir: Path, out_dir: Path | None = None, opset: int = 17) -> Path:
    """HF/sentence-transformers 형식 모델 폴더를 ONNX로 변환한다.

    출력은 last_hidden_state만 내보낸다. mean pooling과 정규화는 onnx 백엔드가 한다.

    Args:
        model_dir: config.json과 가중치가 있는 모델 폴더.
        out_dir: 출력 폴더. None이면 model_dir 안에 쓴다. 다른 폴더면 토크나이저 파일도 복사한다.
        opset: ONNX opset 버전.

    Returns:
        생성된 ONNX 파일 경로 (<out_dir>/onnx/model.onnx).
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError("ONNX 변환에는 extra가 필요합니다: uv sync --extra train") from e

    model_dir = Path(model_dir)
    out_dir = Path(out_dir) if out_dir else model_dir
    onnx_path = onnx_model_path(out_dir)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # sdpa 어텐션은 추적(trace) 시 패딩 분기를 상수로 고정할 수 있어 eager로 변환한다.
    model = AutoModel.from_pretrained(str(model_dir), attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    sample = tokenizer(["query: 예시 문장", "passage: 길이가 다른 예시 문장입니다"], padding=True, return_tensors="pt")
    input_names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    dynamic_axes = {n: {0: "batch", 1: "seq"} for n in input_names}
    dynamic_axes["last_hidden_state"] = {0: "batch", 1: "seq"}

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return self.inner(**dict(zip(input_names, args))).last_hidden_state

    with torch.no_grad():
        torch.onnx.export(
            _Wrapper(model),
            tuple(sample[n] for n in input_names),
            str(onnx_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )

    if out_dir != model_dir:
        _copy_tokenizer(model_dir, out_dir)
    return onnx_path


def _copy_tokenizer(src: Path, dst: Path) -> None:
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


def quantize_onnx(model_dir: Path, out_dir: Path) -> Path:
    """ONNX 모델을 동적 INT8로 양자화해 별도 모델 폴더를 만든다 (extra `[train]` 필요).

    가중치를 INT8로 저장하고 활성값은 실행 중에 양자화한다(동적). 보정 데이터가 필요 없다.
    출력 폴더는 원본과 같은 구성(tokenizer.json + onnx/model.onnx)이라 onnx 백엔드로 그대로 쓴다:
    create_embedding_fn(str(out_dir), backend="onnx"). 폴더 이름이 모델 키가 되어 인덱스도 따로 생긴다.

    Args:
        model_dir: export-onnx를 마친 모델 폴더 (onnx/model.onnx, tokenizer.json).
        out_dir: 출력 폴더 (예: models/multilingual-e5-small-int8).

    Returns:
        양자화한 ONNX 파일 경로 (<out_dir>/onnx/model.onnx).
    """
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    src = onnx_model_path(model_dir)
    if not src.exists():
        raise FileNotFoundError(
            f"ONNX 모델이 없습니다: {src}\n먼저 변환하세요: uv run ragkit export-onnx {model_dir}"
        )
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise ImportError("양자화에는 extra가 필요합니다: uv sync --extra train") from e

    dst = onnx_model_path(out_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    _copy_tokenizer(model_dir, out_dir)
    return dst
