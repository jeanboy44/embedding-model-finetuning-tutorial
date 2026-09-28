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
        for name in TOKENIZER_FILES:
            if (model_dir / name).exists():
                shutil.copy2(model_dir / name, out_dir / name)
    return onnx_path
