"""sentence-transformers 임베딩 백엔드 (extra `[train]` 필요).

모델이 정의한 풀링·Dense·정규화·어댑터를 그대로 쓴다. 비교 모델(EmbeddingGemma)처럼
transformers + 평균 풀링만으로는 맞는 값이 나오지 않는 모델에 쓴다.
입력 형식(앞 문구)은 호출하는 쪽이 모델 프로필로 이미 붙여서 넘긴다.
"""

from pathlib import Path

import numpy as np

from ragkit.embeddings.profiles import ModelProfile
from ragkit.models.embedding_loader import resolve_device, resolve_model_source


def create_st_embedding_fn(
    model_name: str,
    profile: ModelProfile,
    checkpoint_path: Path | None = None,
    device: str | None = None,
):
    """sentence-transformers로 모델을 로드하고 임베딩 생성 함수를 반환한다.

    Args:
        model_name: HuggingFace 모델 이름 또는 로컬 모델 폴더.
        profile: 모델 프로필 (dtype, trust_remote_code, encode 인자).
        checkpoint_path: 파인튜닝한 모델 폴더. 있으면 이 폴더를 읽는다.
        device: "auto" | "cuda" | "mps" | "cpu". None이면 auto.

    Returns:
        texts를 받아 L2 정규화된 (N, dim) np.ndarray를 반환하는 함수.
        (sentence-transformers가 내부에서 길이순으로 배치를 묶는다.)
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("st 백엔드에는 extra가 필요합니다: uv sync --extra train") from e

    source = str(checkpoint_path) if checkpoint_path else resolve_model_source(model_name)
    model_kwargs = {"dtype": getattr(torch, profile.dtype)} if profile.dtype else {}
    model = SentenceTransformer(
        source,
        device=resolve_device(device),
        trust_remote_code=profile.trust_remote_code,
        model_kwargs=model_kwargs,
    )

    def _embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = model.encode(
            texts,
            # 모델의 기본 프롬프트(config의 default_prompt_name)를 끈다.
            # 형식은 프로필(format_query/format_doc)이 이미 붙였으므로, 켜 두면 "Document: Query: ..."가 된다.
            prompt="",
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **profile.encode_kwargs,
        )
        return np.asarray(vectors, dtype=np.float32)

    return _embed
