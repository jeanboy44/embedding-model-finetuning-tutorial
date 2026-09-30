"""비교할 모델 변형(원본 / ONNX / 양자화)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    """벤치마크 대상 하나.

    Attributes:
        name: 표에 쓰는 이름.
        model: 모델 이름 또는 로컬 폴더 (INT8은 `ragkit quantize`가 만든 폴더).
        backend: onnx | torch.
        device: torch 장치. 배포 서버를 흉내 내려고 기본은 cpu.
        deps: 이 변형을 배포할 때 필요한 추론 패키지 (설치 크기 계산용).
    """

    name: str
    model: str
    backend: str
    device: str = "cpu"
    deps: tuple[str, ...] = ()


BASE_MODEL = "models/multilingual-e5-small"

DEFAULT_VARIANTS = (
    Variant("torch-fp32", BASE_MODEL, "torch", deps=("torch", "transformers")),
    Variant("onnx-fp32", BASE_MODEL, "onnx", deps=("onnxruntime", "tokenizers")),
    Variant("onnx-int8", f"{BASE_MODEL}-int8", "onnx", deps=("onnxruntime", "tokenizers")),
    # 어휘 25만 → 약 2만 (ragkit prune-vocab) 후 INT8. 한국어 법령 전용으로 좁힌 모델
    Variant("onnx-int8-pruned", f"{BASE_MODEL}-pruned-int8", "onnx", deps=("onnxruntime", "tokenizers")),
)
