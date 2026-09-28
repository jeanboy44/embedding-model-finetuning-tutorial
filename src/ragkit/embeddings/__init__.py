"""임베딩 생성. 백엔드는 onnx(배포 기본), torch(extra `[torch]`), st(sentence-transformers, extra `[train]`).

모델마다 입력 형식과 지원 백엔드가 다르므로 모델 프로필(`get_profile`)을 함께 쓴다:

    profile = get_profile(model)
    embed = create_embedding_fn(model)                  # 프로필이 지원하는 백엔드로 로드
    q_vec = embed([profile.format_query("주휴수당")])
    d_vec = embed([profile.format_doc(doc)])
"""

from pathlib import Path

from ragkit.config import get_settings
from ragkit.embeddings.prefix import format_passages, format_queries
from ragkit.embeddings.profiles import ModelProfile, get_profile

__all__ = [
    "ModelProfile",
    "choose_backend",
    "create_embedding_fn",
    "format_passages",
    "format_queries",
    "generate_embeddings",
    "get_profile",
    "mean_pool",
]


def choose_backend(profile: ModelProfile, requested: str | None) -> str:
    """사용할 백엔드를 정한다.

    직접 지정하면 그 값을 쓰되 모델이 지원하지 않으면 오류를 낸다. 지정하지 않으면
    Settings.embedding_backend를 쓰고, 모델이 그것을 지원하지 않으면 모델의 첫 번째 백엔드를 쓴다.
    """
    if requested:
        if requested not in profile.backends:
            raise ValueError(
                f"{profile.name}은(는) {requested} 백엔드를 지원하지 않습니다 "
                f"(지원: {', '.join(profile.backends)})"
            )
        return requested
    default = get_settings().embedding_backend
    return default if default in profile.backends else profile.backends[0]


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
    입력 형식(앞 문구)은 붙이지 않는다. 모델 프로필의 format_query / format_doc으로 붙여서 넘긴다.
    속도 옵션의 기본값은 Settings(.env)에서 읽는다.

    Args:
        model_name: HuggingFace 모델 이름 또는 로컬 모델 폴더.
        checkpoint_path: 파인튜닝된 체크포인트 (파일 또는 모델 폴더).
        device: torch/st 백엔드의 장치 "auto" | "cuda" | "mps" | "cpu".
            None이면 Settings.embedding_device (기본 auto: cuda → mps → cpu).
        backend: "onnx" | "torch" | "st". None이면 Settings 값, 모델이 지원하지 않으면 모델 기본값.
        sort_by_length: 길이순 배치로 패딩을 줄인다(onnx/torch). None이면 Settings 값(기본 True).
        num_threads: CPU 연산 스레드 수(onnx/torch). None이면 Settings 값.
        onnx_providers: onnx 실행 공급자 (예: ["CoreMLExecutionProvider", "CPUExecutionProvider"]).

    Returns:
        texts를 받아 L2 정규화된 np.ndarray를 반환하는 함수.

    Example:
        >>> embed = create_embedding_fn("intfloat/multilingual-e5-small")
        >>> doc_embeddings = embed(format_passages(["머신러닝은 AI의 한 분야이다."]))
        >>> query_embedding = embed(format_queries(["머신러닝이란?"]))
    """
    settings = get_settings()
    profile = get_profile(checkpoint_path if checkpoint_path else model_name)
    backend = choose_backend(profile, backend)
    sort_by_length = settings.embedding_sort_by_length if sort_by_length is None else sort_by_length
    num_threads = num_threads or settings.embedding_num_threads
    device = device or settings.embedding_device
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
            model_name, checkpoint_path, device, sort_by_length=sort_by_length, num_threads=num_threads
        )
    if backend == "st":
        from ragkit.embeddings.st_backend import create_st_embedding_fn

        return create_st_embedding_fn(model_name, profile, checkpoint_path, device)
    raise ValueError(f"알 수 없는 임베딩 백엔드: {backend!r} (onnx | torch | st)")


def __getattr__(name: str):
    # torch 백엔드 전용 함수는 쓰는 시점에만 torch를 import한다.
    if name in {"mean_pool", "generate_embeddings"}:
        from ragkit.embeddings import torch_backend

        return getattr(torch_backend, name)
    raise AttributeError(f"module 'ragkit.embeddings' has no attribute {name!r}")
