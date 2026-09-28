"""법령 단위 분할 테스트."""

import json

from ragkit.training.split import load_splits, split_by_law, split_summary, write_splits


def _laws(split, corpus_by_id) -> set[str]:
    return {corpus_by_id[q["positive_id"]]["category"] for q in split}


def test_split_by_law_no_leak_and_sizes(questions, corpus_by_id) -> None:
    """법령이 분할 사이에 겹치지 않는다.

    youth(법령 4개, 질문 16개): test 1 · dev 1 · train 2.
    traffic+electric(법령 3개를 한 그룹으로 합침, 질문 12개): test 1 · dev 1 · train 1.
    """
    splits = split_by_law(questions, corpus_by_id)

    laws = {name: _laws(split, corpus_by_id) for name, split in splits.items()}
    assert not laws["train"] & laws["dev"]
    assert not laws["train"] & laws["test"]
    assert not laws["dev"] & laws["test"]
    assert {name: len(split) for name, split in splits.items()} == {
        "train": 12,
        "dev": 8,
        "test": 8,
    }
    assert sum(len(s) for s in splits.values()) == len(questions)


def test_split_skips_laws_that_overshoot_ratio(questions, corpus_by_id) -> None:
    """질문 수가 크게 다른 법령: 목표를 크게 넘기는 법령은 test·dev에 넣지 않는다.

    마법 질문을 40개로 늘린 그룹(마법 40, 바법 4, 사법 4, 총 48): test 목표 9.6 → 4개짜리 법령 둘(8),
    dev는 마법만 남아(train에 최소 1개) 비고, 가장 큰 마법은 어떤 seed에서도 train.
    """
    small = [q for q in questions if corpus_by_id[q["positive_id"]]["theme"] != "youth"]
    big = [q for q in small if q["positive_id"].startswith("마법_")] * 9  # 4 → 36개 추가
    for seed in range(10):
        splits = split_by_law(small + big, corpus_by_id, seed=seed)
        assert {name: len(s) for name, s in splits.items()} == {"train": 40, "dev": 0, "test": 8}


def test_split_takes_smallest_law_when_all_overshoot(corpus_by_id) -> None:
    """모든 법령이 목표보다 크면 가장 작은 법령 하나를 넣는다 (실제 데이터: 169 · 46 · 44)."""
    sizes = {"마법": 169, "바법": 46, "사법": 44}
    rows = [
        {"query": f"{law} {i}", "positive_id": f"{law}_법률_제2조", "hard_negative_ids": []}
        for law, n in sizes.items()
        for i in range(n)
    ]
    for seed in range(10):
        splits = split_by_law(rows, corpus_by_id, seed=seed)
        assert {name: len(s) for name, s in splits.items()} == {
            "train": 169,
            "dev": 44,
            "test": 46,
        } or {name: len(s) for name, s in splits.items()} == {
            "train": 169,
            "dev": 46,
            "test": 44,
        }


def test_split_does_not_undershoot_ratio(corpus_by_id) -> None:
    """작은 법령만 들어가 test가 목표보다 훨씬 작아지지 않는다 (법령 질문 수 500·400·300·10)."""
    sizes = {"가법": 500, "나법": 400, "다법": 300, "라법": 10}
    rows = [
        {"query": f"{law} {i}", "positive_id": f"{law}_법률_제2조", "hard_negative_ids": []}
        for law, n in sizes.items()
        for i in range(n)
    ]
    for seed in range(10):
        splits = split_by_law(rows, corpus_by_id, seed=seed)
        test_share = len(splits["test"]) / len(rows)
        assert 0.1 <= test_share <= 0.35, (seed, test_share)
        assert splits["train"]


def test_split_by_law_is_reproducible(questions, corpus_by_id) -> None:
    assert split_by_law(questions, corpus_by_id, seed=7) == split_by_law(
        questions, corpus_by_id, seed=7
    )


def test_single_law_stays_in_train(questions, corpus_by_id) -> None:
    """법령이 하나뿐이면 train에 남기고 dev·test는 비운다."""
    only = [q for q in questions if q["positive_id"].startswith("가법_")]

    splits = split_by_law(only, corpus_by_id)

    assert len(splits["train"]) == 4
    assert splits["dev"] == [] and splits["test"] == []


def test_write_and_load_splits(tmp_path, questions, corpus_by_id) -> None:
    splits = split_by_law(questions, corpus_by_id)
    summary = split_summary(splits, corpus_by_id)

    write_splits(splits, tmp_path, {"seed": 42, **summary})

    assert load_splits(tmp_path) == splits
    meta = json.loads((tmp_path / "split_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 42
    assert meta["test"]["questions"] == 8
    assert len(meta["test"]["laws"]) == 2
