"""학습 데이터 준비(분할, 학습 예시)와 파인튜닝.

split·triplets는 core 의존성만 쓴다. 학습(ragkit.training.train)은 [train] extra가 필요하다.
"""

from .inbatch import batch_margins, in_batch_scores
from .split import load_splits, split_by_law, split_summary, write_splits
from .triplets import to_examples

__all__ = [
    "batch_margins",
    "in_batch_scores",
    "load_splits",
    "split_by_law",
    "split_summary",
    "to_examples",
    "write_splits",
]
