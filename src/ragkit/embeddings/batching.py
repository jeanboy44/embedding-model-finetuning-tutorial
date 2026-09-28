"""배치 나누기. 길이순으로 묶으면 배치 안의 패딩(쓸모없는 계산)이 줄어든다."""

from collections.abc import Callable

import numpy as np


def embed_in_batches(
    texts: list[str],
    embed_batch: Callable[[list[str]], np.ndarray],
    *,
    batch_size: int = 32,
    sort_by_length: bool = True,
) -> np.ndarray:
    """texts를 배치로 나눠 embed_batch를 호출하고, 결과를 입력 순서대로 합친다.

    Args:
        texts: 임베딩할 텍스트.
        embed_batch: 텍스트 배치 → (배치, dim) 배열.
        batch_size: 배치 크기.
        sort_by_length: True면 길이가 비슷한 텍스트끼리 묶는다. 배치는 가장 긴 텍스트에
            맞춰 패딩되므로, 길이가 들쭉날쭉한 코퍼스에서 계산량이 크게 준다.

    Returns:
        (len(texts), dim) 배열. 행 순서는 texts와 같다.
    """
    order = np.argsort([len(t) for t in texts], kind="stable") if sort_by_length else np.arange(len(texts))
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(embed_batch([texts[i] for i in order[start : start + batch_size]]))
    stacked = np.concatenate(chunks, axis=0)
    out = np.empty_like(stacked)
    out[order] = stacked
    return out
