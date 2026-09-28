"""임베딩 생성. 백엔드는 onnx(배포 기본) 또는 torch(extra `[torch]`)."""

from pathlib import Path

from ragkit.config import get_settings
from ragkit.embeddings.prefix import format_passages, format_queries

__all__ = [
    "create_embedding_fn",
    "format_passages",
    "format_queries",
    "generate_embeddings",
    "mean_pool",
]


def create_embedding_fn(
    model_name: str,
    checkpoint_path: Path | None = None,
    device: str | None = None,
    backend: str | None = None,
    *,
    sort_by_length: bool | None = None,
    num_threads: int | None = None,
    onnx_providers: list[str] | None = None,
):
    """모델을 로드하고 임베딩 생성 함수를 반환한다.

    팩토리 함수로, 모델 로드를 한 번만 수행하고 이후 호출에서 재사용한다.
    속도 옵션의 기본값은 Settings(.env)에서 읽는다.

    Args:
        model_name: HuggingFace 모델 이름 또는 로컬 모델 폴더.
        checkpoint_path: 파인튜닝된 체크포인트 (파일 또는 모델 폴더).
        device: torch 백엔드의 장치 "auto" | "cuda" | "mps" | "cpu".
            None이면 Settings.embedding_device (기본 auto: cuda → mps → cpu).
        backend: "onnx" 또는 "torch". None이면 Settings.embedding_backend.
        sort_by_length: 길이순 배치로 패딩을 줄인다. None이면 Settings 값(기본 True).
        num_threads: CPU 연산 스레드 수. None이면 Settings 값(기본: 라이브러리 기본값).
        onnx_providers: onnx 실행 공급자 (예: ["CoreMLExecutionProvider", "CPUExecutionProvider"]).

    Returns:
        texts를 받아 L2 정규화된 np.ndarray를 반환하는 함수.

    Example:
        >>> embed = create_embedding_fn("intfloat/multilingual-e5-small")
        >>> doc_embeddings = embed(format_passages(["머신러닝은 AI의 한 분야이다."]))
        >>> query_embedding = embed(format_queries(["머신러닝이란?"]))
    """
    settings = get_settings()
    backend = backend or settings.embedding_backend
    sort_by_length = settings.embedding_sort_by_length if sort_by_length is None else sort_by_length
    num_threads = num_threads or settings.embedding_num_threads
    if backend == "onnx":
        from ragkit.embeddings.onnx_backend import create_onnx_embedding_fn
        from ragkit.models import resolve_model_source

        if checkpoint_path and Path(checkpoint_path).is_dir():
            model_dir = Path(checkpoint_path)
        else:
            model_dir = Path(resolve_model_source(model_name))
        return create_onnx_embedding_fn(
            model_dir, sort_by_length=sort_by_length, num_threads=num_threads, providers=onnx_providers
        )
    if backend == "torch":
        from ragkit.embeddings.torch_backend import create_torch_embedding_fn

        return create_torch_embedding_fn(
            model_name,
            checkpoint_path,
            device or settings.embedding_device,
            sort_by_length=sort_by_length,
            num_threads=num_threads,
        )
    raise ValueError(f"알 수 없는 임베딩 백엔드: {backend!r} (onnx | torch)")


def __getattr__(name: str):
    # torch 백엔드 전용 함수는 쓰는 시점에만 torch를 import한다.
    if name in {"mean_pool", "generate_embeddings"}:
        from ragkit.embeddings import torch_backend

        return getattr(torch_backend, name)
    raise AttributeError(f"module 'ragkit.embeddings' has no attribute {name!r}")
