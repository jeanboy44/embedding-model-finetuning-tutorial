"""학습/평가 분할: 법령 단위 또는 조 단위.

- 법령 단위: 한 법령의 질문은 모두 같은 분할에 들어간다. test 점수는 "학습 때 본 적 없는 법령"에서의 검색 성능이다.
- 조 단위: 한 조(긴 조문의 항 · 호 조각 포함)의 질문은 모두 같은 분할에 들어간다. test 점수는
  "학습한 법령 안의, 학습 질문이 없던 조"에서의 검색 성능이다.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ragkit.data import load_questions, relevance_key

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

    테마마다 법령을 seed로 섞은 순서대로 보면서, 넣으면 test 질문 수가 test_ratio 목표에
    더 가까워지는 법령을 test에 넣는다. 하나도 못 넣었으면 가장 작은 법령 하나를 넣는다.
    dev도 같은 방법으로 채우고 나머지는 train에 넣는다. 그룹마다 train에 법령이
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
            target = ratio * total
            chosen: list[str] = []
            taken = 0
            for law in remaining:
                closer = abs(taken + counts[law] - target) < abs(taken - target)
                if len(remaining) - len(chosen) > 1 and closer:
                    chosen.append(law)
                    taken += counts[law]
            if not chosen and target > 0 and len(remaining) > 1:
                chosen.append(min(remaining, key=lambda law: (counts[law], law)))
            for law in chosen:
                assignment[law] = name
                remaining.remove(law)
        for law in remaining:
            assignment[law] = "train"

    splits: dict[str, list[dict]] = {name: [] for name in SPLITS}
    for question, law in zip(questions, law_of):
        splits[assignment[law]].append(question)
    return splits


def split_by_article(
    questions: list[dict],
    corpus_by_id: dict[str, dict],
    *,
    test_ratio: float = 0.15,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """질문을 조 단위로 train/dev/test에 나눈다. 법령마다 따로 나눠(층화) 모든 법령이 고루 들어간다.

    묶음은 정답 문서의 조(relevance_key: 조각이면 parent_id)다. 법령마다 조를 seed로 섞은 순서대로
    보면서, 넣으면 test 질문 수가 test_ratio 목표에 더 가까워지는 조를 test에 넣는다. dev도 같은
    방법으로 채우고 나머지는 train에 넣는다. 법령마다 train에 조가 최소 1개 남는다.

    Args:
        questions: filter_questions를 거친 질문 목록.
        corpus_by_id: id → 문서. 법령(category)과 조를 positive 문서에서 읽는다.
        test_ratio: 법령별 test 질문 비율 목표.
        dev_ratio: 법령별 dev 질문 비율 목표.
        seed: 조 섞기 seed.

    Returns:
        {"train": [...], "dev": [...], "test": [...]}. 각 목록은 입력 순서를 유지한다.
    """
    positives = [corpus_by_id[q["positive_id"]] for q in questions]
    article_of = [relevance_key(doc) for doc in positives]
    counts = Counter(article_of)
    articles_by_law: dict[str, list[str]] = defaultdict(list)
    for doc, article in sorted(zip(positives, article_of), key=lambda pair: pair[1]):
        if article not in articles_by_law[doc["category"]]:
            articles_by_law[doc["category"]].append(article)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for law in sorted(articles_by_law):
        remaining = list(articles_by_law[law])
        rng.shuffle(remaining)
        total = sum(counts[a] for a in remaining)
        for name, ratio in (("test", test_ratio), ("dev", dev_ratio)):
            target = ratio * total
            taken = 0
            for article in list(remaining):
                closer = abs(taken + counts[article] - target) < abs(taken - target)
                if len(remaining) > 1 and closer:
                    assignment[article] = name
                    remaining.remove(article)
                    taken += counts[article]
        for article in remaining:
            assignment[article] = "train"

    splits: dict[str, list[dict]] = {name: [] for name in SPLITS}
    for question, article in zip(questions, article_of):
        splits[assignment[article]].append(question)
    return splits


def drop_heldout_negatives(
    splits: dict[str, list[dict]], corpus_by_id: dict[str, dict]
) -> tuple[dict[str, list[dict]], dict]:
    """train 질문의 hard negative에서 dev · test 정답 조에 속한 문서를 뺀다.

    조 단위 분할에서는 같은 법령의 다른 조가 hard negative로 붙어 있어, 그대로 두면 test 정답 조가
    학습 때 "오답"으로 나온다. negative가 하나도 안 남은 질문은 배치 안의 다른 정답만 오답으로 쓴다.

    Returns:
        (train을 바꾼 분할, {"dropped_negatives", "train_negatives", "no_negative_questions"}).
    """
    heldout = {
        relevance_key(corpus_by_id[q["positive_id"]])
        for name in ("dev", "test")
        for q in splits[name]
    }
    total = dropped = empty = 0
    train = []
    for q in splits["train"]:
        negatives = q.get("hard_negative_ids") or []
        kept = [n for n in negatives if relevance_key(corpus_by_id[n]) not in heldout]
        total += len(negatives)
        dropped += len(negatives) - len(kept)
        empty += bool(negatives) and not kept
        train.append({**q, "hard_negative_ids": kept})
    stats = {"train_negatives": total, "dropped_negatives": dropped, "no_negative_questions": empty}
    return {**splits, "train": train}, stats


def split_summary(splits: dict[str, list[dict]], corpus_by_id: dict[str, dict]) -> dict:
    """분할별 질문 수, 조 수, 법령 목록."""
    return {
        name: {
            "questions": len(split),
            "articles": len({relevance_key(corpus_by_id[q["positive_id"]]) for q in split}),
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
