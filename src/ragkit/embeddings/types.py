from collections.abc import Callable

import numpy as np

EmbedFn = Callable[..., np.ndarray]
"""텍스트 목록을 받아 (n, dim) L2 정규화 임베딩을 돌려주는 함수. create_embedding_fn이 만든다."""

__all__ = ["EmbedFn"]
