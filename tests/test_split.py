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
