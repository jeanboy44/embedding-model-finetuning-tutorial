"""학습/평가 분할: 법령 단위.

한 법령의 질문은 모두 같은 분할에 들어간다. test 점수는 "학습 때 본 적 없는 법령"에서의 검색 성능이다.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ragkit.data import load_questions

SPLITS = ("train", "dev", "test")
# 법령이 이보다 적은 테마는 따로 나누면 train이 비므로, 그런 테마끼리 한 그룹으로 합쳐 나눈다.
MIN_LAWS_PER_GROUP = 3


def split_by_law(
    questions: list[dict],
    corpus_by_id: dict[str, dict],
    *,
    test_ratio: float = 0.2,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """질문을 법령 단위로 train/dev/test에 나눈다.

    테마마다 법령을 seed로 섞고, 질문 수가 test_ratio에 닿을 때까지 법령을 test에,
    이어서 dev_ratio만큼 dev에 넣고, 나머지는 train에 넣는다. 그룹마다 train에 법령이
    최소 1개 남으므로 법령이 적으면 dev·test가 빌 수 있다.

    Args:
        questions: filter_questions를 거친 질문 목록.
        corpus_by_id: id → 문서. 법령(category)과 테마(theme)를 positive 문서에서 읽는다.
        test_ratio: 그룹별 test 질문 비율 목표.
        dev_ratio: 그룹별 dev 질문 비율 목표.
        seed: 법령 섞기 seed.

    Returns:
        {"train": [...], "dev": [...], "test": [...]}. 각 목록은 입력 순서를 유지한다.
    """
    positives = [corpus_by_id[q["positive_id"]] for q in questions]
    law_of = [doc["category"] for doc in positives]
    counts = Counter(law_of)
    theme_of = {doc["category"]: doc["theme"] for doc in positives}
    laws_by_theme: dict[str, list[str]] = defaultdict(list)
    for law in sorted(counts):
        laws_by_theme[theme_of[law]].append(law)

    groups = [laws for laws in laws_by_theme.values() if len(laws) >= MIN_LAWS_PER_GROUP]
    small = sorted(
        law for laws in laws_by_theme.values() if len(laws) < MIN_LAWS_PER_GROUP for law in laws
    )
    if small:
        groups.append(small)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for group in groups:
        remaining = list(group)
        rng.shuffle(remaining)
        total = sum(counts[law] for law in remaining)
        for name, ratio in (("test", test_ratio), ("dev", dev_ratio)):
            taken = 0
            while len(remaining) > 1 and taken < ratio * total:
                law = remaining.pop(0)
                assignment[law] = name
                taken += counts[law]
        for law in remaining:
            assignment[law] = "train"

    splits: dict[str, list[dict]] = {name: [] for name in SPLITS}
    for question, law in zip(questions, law_of):
        splits[assignment[law]].append(question)
    return splits


def split_summary(splits: dict[str, list[dict]], corpus_by_id: dict[str, dict]) -> dict:
    """분할별 질문 수와 법령 목록."""
    return {
        name: {
            "questions": len(split),
            "laws": sorted({corpus_by_id[q["positive_id"]]["category"] for q in split}),
        }
        for name, split in splits.items()
    }


def write_splits(splits: dict[str, list[dict]], out_dir: Path, meta: dict) -> None:
    """분할을 train/dev/test.jsonl로, 설정·요약을 split_meta.json으로 저장한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        lines = [json.dumps(q, ensure_ascii=False) for q in splits.get(name, [])]
        (out_dir / f"{name}.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    (out_dir / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_splits(splits_dir: Path) -> dict[str, list[dict]]:
    """write_splits로 저장한 분할을 읽는다. 없는 파일은 빈 목록이다."""
    splits_dir = Path(splits_dir)
    return {
        name: load_questions(splits_dir / f"{name}.jsonl")
        if (splits_dir / f"{name}.jsonl").exists()
        else []
        for name in SPLITS
    }
