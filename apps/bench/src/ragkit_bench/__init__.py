"""배포 최적화 벤치마크: 원본(torch) / ONNX / INT8 양자화 모델의 속도·메모리·크기·정확도 비교 (3단계)."""

from .measure import install_size, measure, model_size
from .report import to_markdown
from .variants import DEFAULT_VARIANTS, Variant

__all__ = ["DEFAULT_VARIANTS", "Variant", "install_size", "main", "measure", "model_size", "to_markdown"]


def main() -> None:
    """CLI 진입점 (ragkit-bench)."""
    from .cli import app

    app()
