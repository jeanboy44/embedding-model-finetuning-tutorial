"""같은 법 묶음 배치 샘플러."""

import pytest

pytest.importorskip("sentence_transformers")
from datasets import Dataset

from ragkit.training.sampler import SameLawBatchSampler


def _dataset(n_per_law: dict[str, int]) -> tuple[Dataset, list[str]]:
    rows, groups = [], []
    for law, n in n_per_law.items():
        for i in range(n):
            rows.append({"anchor": f"q {law} {i}", "positive": f"p {law} {i}", "negative_1": f"n {law} {i}"})
            groups.append(law)
    return Dataset.from_list(rows), groups


def test_full_ratio_fills_batch_with_one_law():
    ds, groups = _dataset({"가법": 40, "나법": 40})
    sampler = SameLawBatchSampler(ds, groups=groups, ratio=1.0, batch_size=8, drop_last=False, seed=1)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 10
    assert sorted(i for b in batches for i in b) == list(range(80))
    assert all(len({groups[i] for i in b}) == 1 for b in batches)


def test_half_ratio_and_no_duplicate_texts():
    rows = [{"anchor": f"q{i}", "positive": "같은 정답" if i < 4 else f"p{i}", "negative_1": f"n{i}"} for i in range(32)]
    groups = ["가법"] * 16 + ["나법"] * 16
    stats: dict = {}
    sampler = SameLawBatchSampler(
        Dataset.from_list(rows), groups=groups, ratio=0.5, batch_size=8, drop_last=False, seed=3, stats=stats
    )
    batches = list(sampler)
    for b in batches:
        texts = [rows[i][c] for i in b for c in rows[i]]
        assert len(texts) == len(set(texts))
    assert stats["epoch_0"]["batches"] == len(batches)


def test_seed_and_epoch_change_order():
    ds, groups = _dataset({"가법": 20, "나법": 20})
    first = list(SameLawBatchSampler(ds, groups=groups, ratio=1.0, batch_size=4, drop_last=False, seed=7))
    again = SameLawBatchSampler(ds, groups=groups, ratio=1.0, batch_size=4, drop_last=False, seed=7)
    assert list(again) == first
    again.set_epoch(1)
    assert list(again) != first
