"""같은 법 묶음 배치 샘플러: in-batch negative를 같은 법령 문서로 채운다.

무작위 배치(NO_DUPLICATES)에서는 질문 하나가 보는 in-batch 후보 대부분이 다른 법 문서라
"같은 법 안의 비슷한 조문"을 가르는 신호가 약하다. 이 샘플러는 배치마다 법령 하나를 골라
batch_size × ratio 행을 그 법령 질문으로 채우고, 나머지는 남은 행에서 무작위로 채운다.
NO_DUPLICATES와 같이 한 배치 안에 같은 텍스트(질문 · 정답 · negative)가 두 번 들어가지 않게 한다.

[train] extra가 필요하다. train.py에서만 import한다.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterator

from sentence_transformers.base.sampler import DefaultBatchSampler


class SameLawBatchSampler(DefaultBatchSampler):
    """배치마다 한 법령의 행을 ratio만큼 모으는 배치 샘플러.

    Args:
        dataset: 학습 예시 Dataset (anchor, positive, negative_n 열).
        groups: 행마다 법령 이름 (dataset과 같은 순서).
        ratio: 배치에서 같은 법령으로 채울 비율 (1.0이면 전부, 0.5면 절반).
        stats: 주면 epoch마다 배치 통계를 stats["epoch_<n>"]에 남긴다.
        batch_size, drop_last, valid_label_columns, generator, seed: DefaultBatchSampler와 같다.
    """

    def __init__(
        self,
        dataset,
        groups: list[str],
        ratio: float,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator=None,
        seed: int = 0,
        stats: dict | None = None,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        if len(groups) != len(dataset):
            raise ValueError(
                f"groups 길이({len(groups)})가 dataset 행 수({len(dataset)})와 다릅니다."
            )
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio는 0 초과 1 이하여야 합니다: {ratio}")
        columns = [
            c for c in dataset.column_names if c not in set(valid_label_columns or [])
        ]
        # 행마다 텍스트 집합. 한 배치 안에서 겹치면 가짜 negative가 된다.
        self.row_texts: list[frozenset[str]] = [
            frozenset(dataset[c][i] for c in columns) for i in range(len(dataset))
        ]
        self.groups = groups
        self.ratio = ratio
        self.n_same = max(1, round(batch_size * ratio))
        self.stats = stats

    def _take(
        self,
        candidates: list[int],
        used: set[int],
        seen: set[str],
        batch: list[int],
        limit: int,
    ) -> None:
        """candidates 순서대로, 아직 안 쓴 행 중 텍스트가 겹치지 않는 행을 batch에 limit개까지 넣는다."""
        for idx in candidates:
            if len(batch) >= limit:
                return
            if idx in used or self.row_texts[idx] & seen:
                continue
            batch.append(idx)
            used.add(idx)
            seen |= self.row_texts[idx]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        by_group: dict[str, list[int]] = {}
        for idx, group in enumerate(self.groups):
            by_group.setdefault(group, []).append(idx)
        for members in by_group.values():
            rng.shuffle(members)
        pool = list(range(len(self.groups)))
        rng.shuffle(pool)

        used: set[int] = set()
        same_counts: list[int] = []
        # Trainer가 len()으로 step 수를 정하므로 배치 수를 len()에 맞춘다.
        # 겹침 때문에 끝까지 못 넣은 행(있으면 몇 개)은 이번 epoch에서 빠지고 stats에 남는다.
        while len(used) < len(pool) and len(same_counts) < len(self):
            left = {g: [i for i in m if i not in used] for g, m in by_group.items()}
            left = {g: m for g, m in left.items() if m}
            names = sorted(left)
            # 남은 행이 많은 법령일수록 자주 뽑힌다 → 법령마다 행 수에 비례해 배치를 받는다
            group = rng.choices(names, weights=[len(left[g]) for g in names])[0]
            batch: list[int] = []
            seen: set[str] = set()
            self._take(left[group], used, seen, batch, self.n_same)
            self._take(pool, used, seen, batch, self.batch_size)
            # 겹침 때문에 더 못 넣는 마지막 몇 행은 다음 배치로 넘어간다
            if not batch:
                break
            same_counts.append(
                Counter(self.groups[i] for i in batch).most_common(1)[0][1]
            )
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch
        if self.stats is not None:
            self.stats[f"epoch_{self.epoch}"] = {
                "batches": len(same_counts),
                "mean_top_law_rows": sum(same_counts) / max(1, len(same_counts)),
                "rows": len(used),
                "dropped_rows": len(pool) - len(used),
            }

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.groups) // self.batch_size
        return math.ceil(len(self.groups) / self.batch_size)
