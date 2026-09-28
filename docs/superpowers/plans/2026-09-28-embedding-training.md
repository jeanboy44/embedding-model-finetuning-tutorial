# 임베딩 학습·평가 (ragkit.data / training / evaluation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 법령 질문 데이터로 e5-small을 전체 학습 / LoRA로 파인튜닝하고, 전체 코퍼스 대상 검색 지표로 base와 비교하는 ragkit 모듈과 CLI를 만든다.

**Architecture:** 순수 로직(`ragkit.data`, `ragkit.training.split`, `ragkit.training.triplets`, `ragkit.evaluation`)은 numpy만 쓰고 가짜 데이터로 테스트한다. 학습(`ragkit.training.train`)은 sentence-transformers를 함수 안에서 import한다([train] extra). CLI 하위 명령(`split`/`train`/`evaluate`)은 `ragkit/cli/train_cli.py`에 두고 `cli_tool.py`의 `app`에 등록만 한다.

**Tech Stack:** Python 3.12, numpy, pydantic, sentence-transformers 6.1, peft 0.21, transformers 5.17, datasets 5, cyclopts, pytest, uv workspace

**Spec:** `docs/superpowers/specs/2026-09-28-embedding-training-design.md`

## Global Constraints

- 브랜치 `feat/training-scripts`, 기준 커밋 `676dac2`(feat/ragkit-phase01). 모든 명령은 워크트리 `.claude/worktrees/training-scripts`에서 실행한다.
- 환경: `uv sync --all-packages --all-extras`. 테스트: `uv run pytest`.
- `pyproject.toml`은 수정하지 않는다(ragkit 세션 담당). 새 의존성 없음(필요한 것은 이미 [train] extra에 있음).
- `ragkit.data`, `ragkit.training.split`, `ragkit.training.triplets`, `ragkit.evaluation`은 torch / sentence-transformers / ragkit.embeddings / ragkit.models를 import하지 않는다(core 설치에서도 import 가능해야 함).
- 앞 문구 기본값은 `"query: "` / `"passage: "`. CLI는 `get_settings().query_prefix` / `passage_prefix`를 넘긴다.
- 문서 텍스트는 항상 `ragkit.data.doc_text(doc)` = `title + "\n" + text`.
- 분할은 법령 단위(`category`), 기본 `test_ratio=0.2`, `dev_ratio=0.1`, `seed=42`.
- LoRA 기본값: `r=16`, `alpha=32`, `dropout=0.1`, `target_modules=["query", "key", "value"]`. peft로 감쌀 대상은 `model[0].model`(`auto_model`은 읽기 전용 별칭이라 대입하면 forward에 연결되지 않는다 — 확인함).
- 학습 기본값: batch 32, epoch 1, lr 2e-5, warmup 0.1(transformers 5에서는 `warmup_steps`에 실수로 넘긴다), max_seq_length 512, seed 42.
- 지표 키 이름: `recall@{k}`, `mrr@10`, `ndcg@10`. 판정 이름: `doc`, `article`.
- 코퍼스 임베딩 파일 캐시는 이번 범위에서 뺀다(`evaluate_retrieval`의 `corpus_embeddings` 인자로 재사용 가능).
- 코드 스타일: 기존 파일처럼 한국어 docstring(Google 스타일 Args/Returns), 타입 힌트, `ruff check` 통과.
- 커밋 메시지 끝: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. 같은 조의 다른 조각이 hard negative로 들어온 질문 → 필터에서 지워져야 한다(가짜 negative). Task 1 테스트로 고정.
2. 법령이 1~2개뿐인 테마(실제 테스트 데이터가 테마당 법령 1개) → train이 비지 않고, 작은 테마끼리 합쳐 나뉜다. Task 2 테스트로 고정.
3. `related_ids`가 정답보다 위에 나온 질문 → 순위에서 빠져 정답 순위가 올라가야 한다. Task 4 테스트로 고정.
4. 같은 조의 조각이 상위에 여러 개 나오는 경우 `article` 판정 → 첫 등장만 센다. Task 4 테스트로 고정.
5. LoRA 병합 후 저장한 모델에 peft 흔적이 남아 ragkit torch 백엔드(`from_pretrained`)가 못 읽는 경우 → Task 5 smoke 테스트에서 `create_embedding_fn(..., backend="torch")`로 다시 읽어 확인.

---

## File Structure

| 파일 | 책임 |
|---|---|
| `src/ragkit/data/__init__.py` | 코퍼스·질문 로드, `relevance_key`, `doc_text`, `filter_questions` |
| `src/ragkit/training/__init__.py` | 공개 이름 재노출(무거운 import 없음) |
| `src/ragkit/training/split.py` | 법령 단위 분할, 분할 저장·로드, 요약 |
| `src/ragkit/training/triplets.py` | 질문 → 학습 예시(anchor/positive/negative_n) |
| `src/ragkit/training/train.py` | `TrainConfig`, `LoraSettings`, `load_train_config`, `train` |
| `src/ragkit/evaluation/__init__.py` | `first_rank`, `question_metrics`, `evaluate_retrieval` |
| `src/ragkit/cli/train_cli.py` | CLI 하위 명령 `split`, `train`, `evaluate` |
| `src/ragkit/cli/cli_tool.py` | 위 세 명령을 `app`에 등록(추가 2줄) |
| `experiments/exp_002_finetuned/config.yaml` | 전체 학습 설정 |
| `experiments/exp_004_lora/config.yaml` | LoRA 설정 |
| `tests/conftest.py` | 가짜 코퍼스·질문 fixture, `slow` 마커 등록 |
| `tests/test_data.py`, `tests/test_split.py`, `tests/test_triplets.py`, `tests/test_evaluation.py`, `tests/test_train_smoke.py`, `tests/test_train_cli.py` | 테스트 |

---

### Task 1: ragkit.data (로드, relevance_key, doc_text, filter_questions)

**Files:**
- Modify: `src/ragkit/data/__init__.py` (현재 docstring만 있음 — 전체 교체)
- Create: `tests/conftest.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `load_corpus(path: Path) -> list[dict]`
  - `load_questions(path: Path) -> list[dict]` (파일 또는 폴더)
  - `relevance_key(doc: dict) -> str`
  - `doc_text(doc: dict) -> str`
  - `filter_questions(questions: list[dict], corpus_by_id: dict[str, dict]) -> tuple[list[dict], dict[str, int]]` — 통계 키 `missing_positive`, `dropped_negatives`
  - fixture `corpus`(28개 문서), `corpus_by_id`, `questions`(28개, 문서마다 1개)

- [ ] **Step 1: fixture 작성 (`tests/conftest.py`)**

```python
"""공용 테스트 fixture: 가짜 법령 코퍼스와 질문."""

import pytest

# 테마별 법령. youth는 법령이 4개, traffic·electric은 3개 미만이라 분할 때 한 그룹으로 합쳐진다.
LAWS = {
    "youth": ["가법", "나법", "다법", "라법"],
    "traffic": ["마법"],
    "electric": ["바법", "사법"],
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: 실제 모델을 쓰는 느린 테스트")


def make_doc(law: str, theme: str, article: str, paragraph: str = "") -> dict:
    """prepare_law_data.py 출력과 같은 필드를 가진 가짜 문서."""
    parent = f"{law}_법률_{article}"
    doc_id = f"{parent}_{paragraph}" if paragraph else parent
    return {
        "id": doc_id,
        "title": f"{law} {article} {paragraph}".strip(),
        "text": f"{law} {article} {paragraph} 본문".replace("  ", " "),
        "category": law,
        "theme": theme,
        "law_name": law,
        "parent_id": parent,
        "paragraph": paragraph,
    }


@pytest.fixture
def corpus() -> list[dict]:
    """법령마다 문서 4개: 제1조는 두 조각(제1항, 제2항), 제2조, 제3조."""
    docs = []
    for theme, laws in LAWS.items():
        for law in laws:
            docs.append(make_doc(law, theme, "제1조", "제1항"))
            docs.append(make_doc(law, theme, "제1조", "제2항"))
            docs.append(make_doc(law, theme, "제2조"))
            docs.append(make_doc(law, theme, "제3조"))
    return docs


@pytest.fixture
def corpus_by_id(corpus: list[dict]) -> dict[str, dict]:
    return {doc["id"]: doc for doc in corpus}


@pytest.fixture
def questions(corpus: list[dict]) -> list[dict]:
    """문서마다 질문 1개. hard negative는 같은 법령의 다른 조 문서 2개."""
    rows = []
    for doc in corpus:
        others = [
            d["id"]
            for d in corpus
            if d["category"] == doc["category"] and d["parent_id"] != doc["parent_id"]
        ]
        rows.append(
            {
                "query": f"{doc['id']} 에 대한 질문",
                "positive_id": doc["id"],
                "hard_negative_ids": others[:2],
                "query_type": "situation",
                "answer": "답",
                "related_ids": [],
            }
        )
    return rows
```

- [ ] **Step 2: 실패하는 테스트 작성 (`tests/test_data.py`)**

```python
"""ragkit.data 테스트."""

import json

from ragkit.data import (
    doc_text,
    filter_questions,
    load_corpus,
    load_questions,
    relevance_key,
)


def test_relevance_key_uses_parent_id(corpus_by_id) -> None:
    """조각 문서는 원래 조 id, parent_id가 없으면 자기 id."""
    assert relevance_key(corpus_by_id["가법_법률_제1조_제2항"]) == "가법_법률_제1조"
    assert relevance_key({"id": "x"}) == "x"


def test_doc_text_joins_title_and_text(corpus_by_id) -> None:
    doc = corpus_by_id["가법_법률_제2조"]
    assert doc_text(doc) == "가법 제2조\n가법 제2조 본문"


def test_load_corpus_and_questions(tmp_path, corpus, questions) -> None:
    """파일 하나 또는 폴더(안의 *.jsonl 이름순)를 읽고, 빈 줄은 건너뛴다."""
    corpus_path = tmp_path / "law_docs.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    assert load_corpus(corpus_path) == corpus

    qdir = tmp_path / "questions"
    qdir.mkdir()
    lines = [json.dumps(q, ensure_ascii=False) for q in questions]
    (qdir / "b__p01.jsonl").write_text("\n".join(lines[2:4]) + "\n\n", encoding="utf-8")
    (qdir / "a__p01.jsonl").write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    assert load_questions(qdir) == questions[:4]
    assert load_questions(qdir / "a__p01.jsonl") == questions[:2]


def test_filter_questions_drops_missing_and_bad_negatives(corpus_by_id) -> None:
    """없는 positive는 질문째 빼고, negative에서는 없는 id·related·같은 조 조각을 지운다."""
    questions = [
        {"query": "없는 정답", "positive_id": "없는법_법률_제1조", "hard_negative_ids": []},
        {
            "query": "제1항 질문",
            "positive_id": "가법_법률_제1조_제1항",
            "hard_negative_ids": [
                "가법_법률_제1조_제2항",  # 같은 조의 다른 조각 → 지움
                "가법_법률_제2조",  # related → 지움
                "없는법_법률_제9조",  # 코퍼스에 없음 → 지움
                "가법_법률_제3조",  # 남김
                "가법_법률_제3조",  # 중복 → 하나만 남김
            ],
            "related_ids": ["가법_법률_제2조"],
        },
    ]

    kept, stats = filter_questions(questions, corpus_by_id)

    assert [q["query"] for q in kept] == ["제1항 질문"]
    assert kept[0]["hard_negative_ids"] == ["가법_법률_제3조"]
    assert stats == {"missing_positive": 1, "dropped_negatives": 4}
    # 입력은 바꾸지 않는다
    assert len(questions[1]["hard_negative_ids"]) == 5
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'doc_text' from 'ragkit.data'`

- [ ] **Step 4: 구현 (`src/ragkit/data/__init__.py` 전체 교체)**

```python
"""코퍼스·질문 데이터 로드와 정리.

코퍼스는 scripts/prepare_law_data.py가 만든 law_docs.json(조문 또는 조문 조각 목록),
질문은 law-question-gen 스킬이 만든 JSONL(한 줄에 질문 하나)이다.
"""

import json
from pathlib import Path


def load_corpus(path: Path) -> list[dict]:
    """코퍼스 JSON(문서 목록)을 읽는다.

    Args:
        path: law_docs.json 경로.

    Returns:
        문서 딕셔너리 목록.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_questions(path: Path) -> list[dict]:
    """질문 JSONL을 읽는다. 폴더를 주면 안의 *.jsonl을 이름순으로 모두 읽는다.

    Args:
        path: JSONL 파일 또는 폴더.

    Returns:
        질문 딕셔너리 목록. 빈 줄은 건너뛴다.
    """
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    rows: list[dict] = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def relevance_key(doc: dict) -> str:
    """정답 판정에 쓰는 조 단위 키. 긴 조문의 조각이면 원래 조 id(parent_id)다."""
    return doc.get("parent_id") or doc["id"]


def doc_text(doc: dict) -> str:
    """임베딩할 문서 텍스트. 인덱스·학습·평가가 모두 이 함수를 쓴다."""
    return f"{doc['title']}\n{doc['text']}"


def filter_questions(
    questions: list[dict], corpus_by_id: dict[str, dict]
) -> tuple[list[dict], dict[str, int]]:
    """코퍼스와 맞지 않는 질문과 negative를 걸러 낸다.

    - positive_id가 코퍼스에 없는 질문은 뺀다(옛 코퍼스 id를 쓰는 질문 등).
    - hard_negative_ids에서 코퍼스에 없는 id, related_ids에 있는 id,
      positive와 같은 조(relevance_key)의 조각, 중복을 지운다(가짜 negative 방지).

    Args:
        questions: 질문 목록. 바꾸지 않는다.
        corpus_by_id: id → 문서.

    Returns:
        (남은 질문 목록, {"missing_positive": 뺀 질문 수, "dropped_negatives": 지운 negative 수})
    """
    kept: list[dict] = []
    missing = dropped = 0
    for question in questions:
        positive = corpus_by_id.get(question["positive_id"])
        if positive is None:
            missing += 1
            continue
        related = set(question.get("related_ids") or [])
        key = relevance_key(positive)
        original = question.get("hard_negative_ids") or []
        negatives = [
            neg
            for neg in dict.fromkeys(original)
            if neg in corpus_by_id
            and neg not in related
            and relevance_key(corpus_by_id[neg]) != key
        ]
        dropped += len(original) - len(negatives)
        kept.append({**question, "hard_negative_ids": negatives})
    return kept, {"missing_positive": missing, "dropped_negatives": dropped}
```

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/test_data.py -v && uv run ruff check src/ragkit/data tests/conftest.py tests/test_data.py`
Expected: 4 passed, `All checks passed!`

- [ ] **Step 6: 커밋**

```bash
git add src/ragkit/data/__init__.py tests/conftest.py tests/test_data.py
git commit -m "feat(data): 코퍼스·질문 로드와 질문 필터

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: ragkit.training.split (법령 단위 분할)

**Files:**
- Modify: `src/ragkit/training/__init__.py` (docstring만 있음 — 교체)
- Create: `src/ragkit/training/split.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: fixture `questions`, `corpus_by_id` (Task 1)
- Produces:
  - `split_by_law(questions, corpus_by_id, *, test_ratio=0.2, dev_ratio=0.1, seed=42) -> dict[str, list[dict]]` — 키 `train`, `dev`, `test`
  - `split_summary(splits, corpus_by_id) -> dict[str, dict]` — `{"train": {"questions": int, "laws": list[str]}, ...}`
  - `write_splits(splits, out_dir: Path, meta: dict) -> None` — `train.jsonl`, `dev.jsonl`, `test.jsonl`, `split_meta.json`
  - `load_splits(splits_dir: Path) -> dict[str, list[dict]]` — 없는 파일은 빈 목록
  - 상수 `SPLITS = ("train", "dev", "test")`, `MIN_LAWS_PER_GROUP = 3`

- [ ] **Step 1: 실패하는 테스트 (`tests/test_split.py`)**

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragkit.training.split'`

- [ ] **Step 3: 구현 (`src/ragkit/training/split.py`)**

```python
"""학습/평가 분할: 법령 단위.

한 법령의 질문은 모두 같은 분할에 들어간다. test 점수는 "학습 때 본 적 없는 법령"에서의 검색 성능이다.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

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
    law_of = [corpus_by_id[q["positive_id"]]["category"] for q in questions]
    counts = Counter(law_of)
    theme_of = {
        corpus_by_id[q["positive_id"]]["category"]: corpus_by_id[q["positive_id"]]["theme"]
        for q in questions
    }
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
    from ragkit.data import load_questions

    splits_dir = Path(splits_dir)
    return {
        name: load_questions(splits_dir / f"{name}.jsonl")
        if (splits_dir / f"{name}.jsonl").exists()
        else []
        for name in SPLITS
    }
```

- [ ] **Step 4: `src/ragkit/training/__init__.py` 교체**

```python
"""학습 데이터 준비(분할, 학습 예시)와 파인튜닝.

split·triplets는 core 의존성만 쓴다. 학습(ragkit.training.train)은 [train] extra가 필요하다.
"""

from .split import load_splits, split_by_law, split_summary, write_splits

__all__ = ["load_splits", "split_by_law", "split_summary", "write_splits"]
```

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/test_split.py -v && uv run ruff check src/ragkit/training tests/test_split.py`
Expected: 4 passed, `All checks passed!`

- [ ] **Step 6: 커밋**

```bash
git add src/ragkit/training/__init__.py src/ragkit/training/split.py tests/test_split.py
git commit -m "feat(training): 법령 단위 학습/평가 분할

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: ragkit.training.triplets (학습 예시)

**Files:**
- Create: `src/ragkit/training/triplets.py`
- Modify: `src/ragkit/training/__init__.py` (재노출 추가)
- Test: `tests/test_triplets.py`

**Interfaces:**
- Consumes: `ragkit.data.doc_text`; fixture `questions`, `corpus_by_id`
- Produces: `to_examples(questions, corpus_by_id, *, num_negatives=1, query_prefix="query: ", passage_prefix="passage: ") -> list[dict]` — 키 `anchor`, `positive`, `negative_1` … `negative_{n}`

- [ ] **Step 1: 실패하는 테스트 (`tests/test_triplets.py`)**

```python
"""학습 예시 변환 테스트."""

from ragkit.data import doc_text
from ragkit.training.triplets import to_examples


def test_to_examples_adds_prefixes(questions, corpus_by_id) -> None:
    question = questions[0]

    [row] = to_examples([question], corpus_by_id)

    positive = corpus_by_id[question["positive_id"]]
    negative = corpus_by_id[question["hard_negative_ids"][0]]
    assert row == {
        "anchor": "query: " + question["query"],
        "positive": "passage: " + doc_text(positive),
        "negative_1": "passage: " + doc_text(negative),
    }


def test_to_examples_repeats_short_negatives_and_skips_empty(corpus_by_id) -> None:
    """negative가 모자라면 있는 것을 반복하고, 0개면 그 질문을 건너뛴다."""
    questions = [
        {"query": "하나", "positive_id": "가법_법률_제2조", "hard_negative_ids": ["가법_법률_제3조"]},
        {"query": "없음", "positive_id": "가법_법률_제2조", "hard_negative_ids": []},
    ]

    rows = to_examples(questions, corpus_by_id, num_negatives=3, query_prefix="", passage_prefix="")

    assert len(rows) == 1
    expected = doc_text(corpus_by_id["가법_법률_제3조"])
    assert [rows[0][f"negative_{i}"] for i in (1, 2, 3)] == [expected] * 3
    assert rows[0]["anchor"] == "하나"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_triplets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragkit.training.triplets'`

- [ ] **Step 3: 구현 (`src/ragkit/training/triplets.py`)**

```python
"""질문 → 대조 학습 예시 (anchor, positive, negative_1..n).

MultipleNegativesRankingLoss는 같은 배치의 다른 positive를 자동으로 negative로 쓰고(in-batch negative),
negative_n 열의 hard negative를 추가 후보로 쓴다.
"""

from ragkit.data import doc_text


def to_examples(
    questions: list[dict],
    corpus_by_id: dict[str, dict],
    *,
    num_negatives: int = 1,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
) -> list[dict]:
    """질문을 학습 예시 행으로 바꾼다.

    모든 행의 열 수가 같아야 하므로 hard negative가 num_negatives보다 적으면 있는 것을 반복하고,
    하나도 없으면 그 질문을 건너뛴다.

    Args:
        questions: filter_questions를 거친 질문 목록.
        corpus_by_id: id → 문서.
        num_negatives: 행마다 넣을 hard negative 수.
        query_prefix: 질문 앞 문구 (e5: "query: ").
        passage_prefix: 문서 앞 문구 (e5: "passage: ").

    Returns:
        {"anchor", "positive", "negative_1", ...} 딕셔너리 목록.
    """
    rows: list[dict] = []
    for question in questions:
        negatives = question["hard_negative_ids"]
        if not negatives:
            continue
        row = {
            "anchor": query_prefix + question["query"],
            "positive": passage_prefix + doc_text(corpus_by_id[question["positive_id"]]),
        }
        for i in range(num_negatives):
            negative = corpus_by_id[negatives[i % len(negatives)]]
            row[f"negative_{i + 1}"] = passage_prefix + doc_text(negative)
        rows.append(row)
    return rows
```

- [ ] **Step 4: `src/ragkit/training/__init__.py`에 재노출 추가**

```python
"""학습 데이터 준비(분할, 학습 예시)와 파인튜닝.

split·triplets는 core 의존성만 쓴다. 학습(ragkit.training.train)은 [train] extra가 필요하다.
"""

from .split import load_splits, split_by_law, split_summary, write_splits
from .triplets import to_examples

__all__ = ["load_splits", "split_by_law", "split_summary", "to_examples", "write_splits"]
```

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/test_triplets.py -v && uv run ruff check src/ragkit/training tests/test_triplets.py`
Expected: 2 passed, `All checks passed!`

- [ ] **Step 6: 커밋**

```bash
git add src/ragkit/training/__init__.py src/ragkit/training/triplets.py tests/test_triplets.py
git commit -m "feat(training): 질문을 대조 학습 예시로 변환

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: ragkit.evaluation (검색 지표)

**Files:**
- Modify: `src/ragkit/evaluation/__init__.py` (docstring만 있음 — 전체 교체)
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `ragkit.data.doc_text`, `ragkit.data.relevance_key`
- Produces:
  - `first_rank(ranked_keys: Sequence[str], positive_key: str) -> int | None` — 중복 키는 첫 등장만 센다
  - `question_metrics(rank: int | None, ks: Sequence[int]) -> dict[str, float]` — `recall@{k}`, `mrr@10`, `ndcg@10`
  - `evaluate_retrieval(embed_fn, corpus, questions, *, ks=(1, 5, 10), query_prefix="query: ", passage_prefix="passage: ", corpus_embeddings=None) -> dict` — 키 `n`, `doc`, `article`, `by_query_type`, `by_theme`, `per_question`
  - Phase 1 튜토리얼(ragkit 세션)도 이 시그니처를 쓴다. 바꾸지 않는다.

- [ ] **Step 1: 실패하는 테스트 (`tests/test_evaluation.py`)**

```python
"""검색 평가 지표 테스트."""

import math

import numpy as np
import pytest

from ragkit.data import doc_text
from ragkit.evaluation import evaluate_retrieval, first_rank, question_metrics


def test_first_rank_counts_first_occurrence() -> None:
    assert first_rank(["a", "b", "a", "c"], "c") == 3
    assert first_rank(["a", "b"], "z") is None


def test_question_metrics() -> None:
    assert question_metrics(3, (1, 5)) == {
        "recall@1": 0.0,
        "recall@5": 1.0,
        "mrr@10": pytest.approx(1 / 3),
        "ndcg@10": pytest.approx(1 / math.log2(4)),
    }
    assert question_metrics(None, (1,)) == {"recall@1": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}
    assert question_metrics(11, (1,))["mrr@10"] == 0.0


def _doc(doc_id: str, parent: str, theme: str) -> dict:
    return {"id": doc_id, "title": doc_id, "text": "본문", "parent_id": parent, "theme": theme}


def test_evaluate_retrieval_doc_article_and_related() -> None:
    """q1: 정답 A_1이 3위(doc)지만 같은 조 A_2가 1위라 article로는 1위.
    q2: related B가 1위지만 순위에서 빠져 정답 C가 1위.
    """
    corpus = [
        _doc("A_1", "A", "youth"),
        _doc("A_2", "A", "youth"),
        _doc("B", "B", "youth"),
        _doc("C", "C", "traffic"),
    ]
    vectors = {
        "passage: " + doc_text(corpus[0]): [1, 0, 0, 0],
        "passage: " + doc_text(corpus[1]): [0, 1, 0, 0],
        "passage: " + doc_text(corpus[2]): [0, 0, 1, 0],
        "passage: " + doc_text(corpus[3]): [0, 0, 0, 1],
        "query: q1": [0.1, 0.9, 0.5, 0.0],
        "query: q2": [0.0, 0.0, 0.9, 0.4],
    }

    def embed(texts: list[str]) -> np.ndarray:
        out = np.array([vectors[t] for t in texts], dtype=float)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    questions = [
        {"query": "q1", "positive_id": "A_1", "query_type": "situation"},
        {"query": "q2", "positive_id": "C", "query_type": "keyword", "related_ids": ["B"]},
    ]

    result = evaluate_retrieval(embed, corpus, questions, ks=(1, 5))

    assert result["n"] == 2
    assert result["doc"] == {
        "recall@1": 0.5,
        "recall@5": 1.0,
        "mrr@10": pytest.approx((1 / 3 + 1) / 2),
        "ndcg@10": pytest.approx((0.5 + 1) / 2),
    }
    assert result["article"]["recall@1"] == 1.0
    assert result["by_query_type"]["situation"]["doc"]["recall@1"] == 0.0
    assert result["by_theme"]["traffic"]["n"] == 1
    assert [(r["doc_rank"], r["article_rank"]) for r in result["per_question"]] == [(3, 1), (1, 1)]


def test_evaluate_retrieval_reuses_corpus_embeddings() -> None:
    """corpus_embeddings를 주면 코퍼스를 다시 임베딩하지 않는다."""
    corpus = [_doc("A", "A", "t"), _doc("B", "B", "t")]
    seen: list[str] = []

    def embed(texts: list[str]) -> np.ndarray:
        seen.extend(texts)
        return np.array([[1.0, 0.0]] * len(texts))

    evaluate_retrieval(
        embed,
        corpus,
        [{"query": "q", "positive_id": "A"}],
        corpus_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    assert seen == ["query: q"]


def test_evaluate_retrieval_rejects_bad_input() -> None:
    corpus = [_doc("A", "A", "t")]
    embed = lambda texts: np.ones((len(texts), 1))  # noqa: E731
    with pytest.raises(ValueError, match="평가할 질문"):
        evaluate_retrieval(embed, corpus, [])
    with pytest.raises(ValueError, match="코퍼스에 없는"):
        evaluate_retrieval(embed, corpus, [{"query": "q", "positive_id": "Z"}])
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_retrieval' from 'ragkit.evaluation'`

- [ ] **Step 3: 구현 (`src/ragkit/evaluation/__init__.py` 전체 교체)**

```python
"""검색 평가: 전체 코퍼스에서 질문마다 정답 문서의 순위를 구해 Recall@k, MRR@10, nDCG@10을 낸다.

임베딩 함수(embed_fn)만 받으므로 onnx / torch 백엔드 어느 쪽이든 쓸 수 있다.
판정은 두 가지다.
- doc: 정답 문서 id와 정확히 일치
- article: 같은 조(relevance_key)의 조각이면 정답. 같은 조의 조각이 여러 개 나오면 첫 등장만 센다
related_ids(정답을 부분적으로 담은 문서)는 두 판정 모두에서 순위에서 뺀다.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np

from ragkit.data import doc_text, relevance_key

CUTOFF = 10  # MRR, nDCG를 계산하는 순위 한도
CANDIDATES = 100  # 질문마다 정렬할 상위 문서 수 (article 판정의 중복 제거 여유 포함)


def first_rank(ranked_keys: Sequence[str], positive_key: str) -> int | None:
    """중복 키는 첫 등장만 세어 정답 키의 순위(1부터)를 구한다. 없으면 None."""
    seen: set[str] = set()
    for key in ranked_keys:
        if key in seen:
            continue
        seen.add(key)
        if key == positive_key:
            return len(seen)
    return None


def question_metrics(rank: int | None, ks: Sequence[int]) -> dict[str, float]:
    """정답이 하나인 질문의 지표. Recall@k는 정답이 k위 안에 있으면 1."""
    metrics = {f"recall@{k}": float(rank is not None and rank <= k) for k in ks}
    within = rank is not None and rank <= CUTOFF
    metrics[f"mrr@{CUTOFF}"] = 1.0 / rank if within else 0.0
    metrics[f"ndcg@{CUTOFF}"] = 1.0 / math.log2(rank + 1) if within else 0.0
    return metrics


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "doc": _mean([row["doc"] for row in rows]),
        "article": _mean([row["article"] for row in rows]),
    }


def _group(rows: list[dict], field: str) -> dict[str, dict]:
    return {
        value: _summarize([row for row in rows if row[field] == value])
        for value in sorted({row[field] for row in rows})
    }


def evaluate_retrieval(
    embed_fn: Callable[[list[str]], np.ndarray],
    corpus: list[dict],
    questions: list[dict],
    *,
    ks: Sequence[int] = (1, 5, 10),
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
    corpus_embeddings: np.ndarray | None = None,
) -> dict:
    """전체 코퍼스 대상 검색 성능을 잰다.

    Args:
        embed_fn: 텍스트 목록 → L2 정규화된 (N, dim) 배열.
        corpus: 문서 목록 (law_docs.json).
        questions: query, positive_id(, related_ids, query_type)를 가진 질문 목록.
        ks: Recall@k의 k 값들.
        query_prefix: 질문 앞 문구.
        passage_prefix: 문서 앞 문구.
        corpus_embeddings: 미리 계산한 코퍼스 임베딩 (corpus와 같은 순서). None이면 계산한다.

    Returns:
        {"n", "doc": 지표, "article": 지표, "by_query_type": {유형: {"n", "doc", "article"}},
         "by_theme": {...}, "per_question": [{"query", "positive_id", "doc_rank", "article_rank"}]}

    Raises:
        ValueError: 질문이 없거나, positive_id가 코퍼스에 없는 질문이 있을 때.
    """
    if not questions:
        raise ValueError("평가할 질문이 없습니다.")
    by_id = {doc["id"]: doc for doc in corpus}
    missing = [q["positive_id"] for q in questions if q["positive_id"] not in by_id]
    if missing:
        raise ValueError(
            f"코퍼스에 없는 positive_id가 {len(missing)}개 있습니다 (예: {missing[0]}). "
            "ragkit.data.filter_questions로 먼저 거르세요."
        )

    if corpus_embeddings is None:
        corpus_embeddings = embed_fn([passage_prefix + doc_text(doc) for doc in corpus])
    query_embeddings = embed_fn([query_prefix + q["query"] for q in questions])
    scores = query_embeddings @ corpus_embeddings.T
    depth = min(CANDIDATES, len(corpus))
    top = np.argpartition(-scores, depth - 1, axis=1)[:, :depth]

    ids = [doc["id"] for doc in corpus]
    keys = [relevance_key(doc) for doc in corpus]
    rows: list[dict] = []
    for question, candidates, row_scores in zip(questions, top, scores):
        order = candidates[np.argsort(-row_scores[candidates], kind="stable")]
        ignore = set(question.get("related_ids") or [])
        order = [i for i in order if ids[i] not in ignore]
        positive = by_id[question["positive_id"]]
        doc_rank = first_rank([ids[i] for i in order], positive["id"])
        article_rank = first_rank([keys[i] for i in order], relevance_key(positive))
        rows.append(
            {
                "query": question["query"],
                "positive_id": positive["id"],
                "query_type": question.get("query_type") or "unknown",
                "theme": positive.get("theme") or "unknown",
                "doc_rank": doc_rank,
                "article_rank": article_rank,
                "doc": question_metrics(doc_rank, ks),
                "article": question_metrics(article_rank, ks),
            }
        )

    result = _summarize(rows)
    result["by_query_type"] = _group(rows, "query_type")
    result["by_theme"] = _group(rows, "theme")
    result["per_question"] = [
        {key: row[key] for key in ("query", "positive_id", "doc_rank", "article_rank")}
        for row in rows
    ]
    return result
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_evaluation.py -v && uv run ruff check src/ragkit/evaluation tests/test_evaluation.py`
Expected: 5 passed, `All checks passed!`

- [ ] **Step 5: core만으로 import되는지 확인**

Run: `uv run python -c "import sys; import ragkit.evaluation, ragkit.training, ragkit.data; print('torch' in sys.modules)"`
Expected: `False`

- [ ] **Step 6: 커밋**

```bash
git add src/ragkit/evaluation/__init__.py tests/test_evaluation.py
git commit -m "feat(evaluation): 전체 코퍼스 대상 Recall@k·MRR·nDCG (doc/article 판정)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: ragkit.training.train (전체 학습 / LoRA)

**Files:**
- Create: `src/ragkit/training/train.py`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `to_examples` (Task 3), `ragkit.data.doc_text`, `ragkit.models.resolve_model_source`, `ragkit.embeddings.create_embedding_fn(model_name, checkpoint_path=None, device=None, backend=None)` (ragkit 세션 제공), `evaluate_retrieval` (Task 4)
- Produces:
  - `class LoraSettings(BaseModel)`: `r: int = 16`, `alpha: int = 32`, `dropout: float = 0.1`, `target_modules: list[str] = ["query", "key", "value"]`, `save_adapter: bool = False`
  - `class TrainConfig(BaseModel)`: `model: str = "intfloat/multilingual-e5-small"`, `output_dir: Path`, `batch_size: int = 32`, `epochs: float = 1.0`, `lr: float = 2e-5`, `warmup_ratio: float = 0.1`, `max_seq_length: int = 512`, `num_negatives: int = 1`, `seed: int = 42`, `max_steps: int | None = None`, `limit: int | None = None`, `lora: LoraSettings | None = None`
  - `load_train_config(path: Path) -> TrainConfig` — YAML의 `training:` 블록
  - `train(config, train_questions, dev_questions, corpus, *, query_prefix="query: ", passage_prefix="passage: ") -> dict` — `train_meta.json` 내용을 반환: `seconds`, `total_params`, `trainable_params`, `train_examples`, `dev_before`, `dev_after`, `config`
- 저장 결과: `config.output_dir`에 sentence-transformers 폴더(루트에 `config.json`, `model.safetensors`, `tokenizer.json`). `save_adapter`면 `output_dir/adapter/`.

- [ ] **Step 1: 실패하는 smoke 테스트 (`tests/test_train_smoke.py`)**

```python
"""전체 학습 / LoRA smoke 테스트: 몇 step 학습 → 저장 → ragkit torch 백엔드로 다시 읽어 평가."""

import json
import os
from pathlib import Path

import pytest

MODEL = Path(os.environ.get("RAGKIT_TEST_MODEL", "models/multilingual-e5-small"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (MODEL / "config.json").exists(),
        reason=f"로컬 모델 없음: {MODEL} (RAGKIT_TEST_MODEL로 지정)",
    ),
]


@pytest.mark.parametrize("use_lora", [False, True], ids=["full", "lora"])
def test_train_saves_loadable_model(tmp_path, corpus, questions, use_lora) -> None:
    from safetensors import safe_open

    from ragkit.embeddings import create_embedding_fn
    from ragkit.evaluation import evaluate_retrieval
    from ragkit.training.train import LoraSettings, TrainConfig, train

    out = tmp_path / "model"
    config = TrainConfig(
        model=str(MODEL),
        output_dir=out,
        batch_size=4,
        max_steps=2,
        max_seq_length=64,
        lora=LoraSettings(r=4, alpha=8, save_adapter=True) if use_lora else None,
    )

    meta = train(config, questions[:16], questions[16:20], corpus)

    assert json.loads((out / "train_meta.json").read_text(encoding="utf-8"))["seconds"] > 0
    assert meta["train_examples"] == 16
    assert meta["dev_before"] and meta["dev_after"]
    if use_lora:
        assert meta["trainable_params"] < meta["total_params"] / 10
        assert (out / "adapter" / "adapter_config.json").exists()
    else:
        assert meta["trainable_params"] == meta["total_params"]

    with safe_open(str(out / "model.safetensors"), "pt") as weights:
        assert not any("lora" in key for key in weights.keys())

    embed = create_embedding_fn(str(out), checkpoint_path=out, backend="torch", device="cpu")
    result = evaluate_retrieval(embed, corpus, questions[20:24])
    assert result["n"] == 4


def test_load_train_config(tmp_path) -> None:
    from ragkit.training.train import load_train_config

    path = tmp_path / "config.yaml"
    path.write_text(
        "name: lora\ntraining:\n  output_dir: models/finetuned/x\n  lora:\n    r: 8\n",
        encoding="utf-8",
    )

    config = load_train_config(path)

    assert config.output_dir == Path("models/finetuned/x")
    assert config.lora is not None and config.lora.r == 8 and config.lora.alpha == 32
    assert config.batch_size == 32
```

- [ ] **Step 2: 실패 확인**

Run: `RAGKIT_TEST_MODEL=/Users/jeanboy/workspace/embedding-model-finetuning-tutorial/models/multilingual-e5-small uv run pytest tests/test_train_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragkit.training.train'`

- [ ] **Step 3: 구현 (`src/ragkit/training/train.py`)**

```python
"""임베딩 파인튜닝: sentence-transformers + MultipleNegativesRankingLoss (전체 학습 / LoRA).

[train] extra가 필요하다(uv sync --extra train). 무거운 import는 함수 안에서 한다.
저장 결과는 sentence-transformers 폴더라 ragkit torch 백엔드와 `ragkit export-onnx`가 그대로 읽는다.
LoRA는 학습 후 원래 가중치에 합쳐(merge) 같은 형식으로 저장한다.
"""

import json
import time
from pathlib import Path

from pydantic import BaseModel

from ragkit.data import doc_text
from ragkit.training.triplets import to_examples

INSTALL_HINT = "학습에는 [train] extra가 필요합니다: uv sync --extra train"


class LoraSettings(BaseModel):
    """LoRA 설정. attention의 query/key/value에 저랭크 행렬을 더해 그것만 학습한다."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] = ["query", "key", "value"]
    save_adapter: bool = False


class TrainConfig(BaseModel):
    """학습 설정. 실험 config.yaml의 training: 블록과 같은 모양이다."""

    model: str = "intfloat/multilingual-e5-small"
    output_dir: Path
    batch_size: int = 32
    epochs: float = 1.0
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    num_negatives: int = 1
    seed: int = 42
    max_steps: int | None = None
    limit: int | None = None
    lora: LoraSettings | None = None


def load_train_config(path: Path) -> TrainConfig:
    """실험 config.yaml의 training: 블록을 읽는다."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainConfig(**data["training"])


def _dev_evaluator(dev_questions, corpus, query_prefix, passage_prefix):
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    return InformationRetrievalEvaluator(
        queries={str(i): query_prefix + q["query"] for i, q in enumerate(dev_questions)},
        corpus={doc["id"]: passage_prefix + doc_text(doc) for doc in corpus},
        relevant_docs={str(i): {q["positive_id"]} for i, q in enumerate(dev_questions)},
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        map_at_k=[10],
        name="dev",
        write_csv=False,
        batch_size=64,
    )


def train(
    config: TrainConfig,
    train_questions: list[dict],
    dev_questions: list[dict],
    corpus: list[dict],
    *,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
) -> dict:
    """모델을 학습하고 config.output_dir에 저장한다.

    Args:
        config: 학습 설정.
        train_questions: filter_questions를 거친 학습 질문.
        dev_questions: 학습 전·후 점수를 볼 dev 질문. 비어 있으면 건너뛴다.
        corpus: 전체 코퍼스 (dev 평가 대상, 문서 텍스트 조회).
        query_prefix: 질문 앞 문구.
        passage_prefix: 문서 앞 문구.

    Returns:
        train_meta.json에 저장한 내용.

    Raises:
        ImportError: [train] extra가 없을 때.
        ValueError: 학습할 예시가 없을 때.
    """
    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from sentence_transformers.training_args import BatchSamplers
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    from ragkit.models import resolve_model_source

    corpus_by_id = {doc["id"]: doc for doc in corpus}
    questions = train_questions[: config.limit] if config.limit else train_questions
    examples = to_examples(
        questions,
        corpus_by_id,
        num_negatives=config.num_negatives,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    if not examples:
        raise ValueError("학습할 예시가 없습니다. hard negative가 있는 train 질문이 필요합니다.")

    model_source = config.model if Path(config.model).exists() else resolve_model_source(config.model)
    model = SentenceTransformer(model_source)
    model.max_seq_length = config.max_seq_length
    total_params = sum(p.numel() for p in model.parameters())

    if config.lora:
        from peft import LoraConfig, get_peft_model

        # auto_model은 읽기 전용 별칭이다. forward가 쓰는 실제 모듈(model)을 감싸야 LoRA가 적용된다.
        model[0].model = get_peft_model(
            model[0].model,
            LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
            ),
        )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    evaluator = (
        _dev_evaluator(dev_questions, corpus, query_prefix, passage_prefix) if dev_questions else None
    )
    dev_before = evaluator(model) if evaluator else None

    output_dir = Path(config.output_dir)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps or -1,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        warmup_steps=config.warmup_ratio,  # transformers 5: 1 미만 실수는 전체 step 대비 비율
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # 같은 문서가 한 배치에 두 번 → 가짜 negative 방지
        seed=config.seed,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(examples),
        loss=losses.MultipleNegativesRankingLoss(model),
    )
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    dev_after = evaluator(model) if evaluator else None

    if config.lora:
        if config.lora.save_adapter:
            model[0].model.save_pretrained(str(output_dir / "adapter"))
        model[0].model = model[0].model.merge_and_unload()
    model.save(str(output_dir))

    meta = {
        "seconds": seconds,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_examples": len(examples),
        "dev_before": dev_before,
        "dev_after": dev_after,
        "config": config.model_dump(mode="json"),
    }
    (output_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
```

- [ ] **Step 4: 통과 확인**

Run: `RAGKIT_TEST_MODEL=/Users/jeanboy/workspace/embedding-model-finetuning-tutorial/models/multilingual-e5-small uv run pytest tests/test_train_smoke.py -v && uv run ruff check src/ragkit/training tests/test_train_smoke.py`
Expected: 3 passed, `All checks passed!`
실패 시 확인할 것: `warmup_steps`에 실수를 받지 않는다는 오류가 나면 `warmup_ratio=config.warmup_ratio`로 바꾼다. `dev_before`가 빈 딕셔너리면 evaluator 반환값을 출력해 키를 확인한다.

- [ ] **Step 5: 기본 테스트는 빠르게 도는지 확인**

Run: `uv run pytest -q`
Expected: 모두 통과(`RAGKIT_TEST_MODEL` 없이 워크트리에서 실행하면 smoke 2개는 skip)

- [ ] **Step 6: 커밋**

```bash
git add src/ragkit/training/train.py tests/test_train_smoke.py
git commit -m "feat(training): sentence-transformers 학습 (전체 / LoRA 병합 저장)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: CLI `split` / `train` / `evaluate` + 실험 설정

**Files:**
- Create: `src/ragkit/cli/train_cli.py`
- Modify: `src/ragkit/cli/cli_tool.py` (import 1줄 + 등록 1줄, `app = cyclopts.App(...)` 바로 아래)
- Create: `experiments/exp_002_finetuned/config.yaml`, `experiments/exp_004_lora/config.yaml`
- Test: `tests/test_train_cli.py`

**Interfaces:**
- Consumes: Task 1~5 전부, `ragkit.config.get_settings()` (`data_dir`, `query_prefix`, `passage_prefix`), `ragkit.embeddings.create_embedding_fn`
- Produces:
  - `split(questions: Path, *, corpus: Path | None = None, out: Path | None = None, seed: int = 42, test_ratio: float = 0.2, dev_ratio: float = 0.1, strict: bool = False) -> None`
  - `train(config: Path, *, splits: Path | None = None, corpus: Path | None = None, model: str | None = None, max_steps: int | None = None, limit: int | None = None) -> None`
  - `evaluate(model: str, *, split: str = "test", splits: Path | None = None, corpus: Path | None = None, backend: str = "torch", out: Path | None = None) -> None`
  - 오류 시 `SystemExit(1)`과 안내 메시지

- [ ] **Step 1: 실패하는 테스트 (`tests/test_train_cli.py`)**

```python
"""학습 CLI 테스트 (모델 없이 도는 부분)."""

import json

import numpy as np
import pytest

from ragkit.cli import train_cli


def _write_inputs(tmp_path, corpus, questions):
    corpus_path = tmp_path / "law_docs.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    qpath = tmp_path / "questions.jsonl"
    extra = {"query": "옛 코퍼스", "positive_id": "없는법_법률_제1조", "hard_negative_ids": []}
    lines = [json.dumps(q, ensure_ascii=False) for q in [*questions, extra]]
    qpath.write_text("\n".join(lines), encoding="utf-8")
    return corpus_path, qpath


def test_split_command_writes_splits(tmp_path, corpus, questions, capsys) -> None:
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)

    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")

    meta = json.loads((tmp_path / "splits" / "split_meta.json").read_text(encoding="utf-8"))
    assert meta["train"]["questions"] + meta["dev"]["questions"] + meta["test"]["questions"] == 28
    assert meta["filter"]["missing_positive"] == 1
    assert "뺀 질문 1개" in capsys.readouterr().out


def test_split_command_strict_fails(tmp_path, corpus, questions) -> None:
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    with pytest.raises(SystemExit):
        train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits", strict=True)


def test_evaluate_command_writes_result(tmp_path, corpus, questions, monkeypatch) -> None:
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")

    def fake_create(model_name, checkpoint_path=None, device=None, backend=None):
        return lambda texts: np.ones((len(texts), 2)) / np.sqrt(2)

    monkeypatch.setattr(train_cli, "create_embedding_fn", fake_create)
    out = tmp_path / "result.json"

    train_cli.evaluate("base-model", splits=tmp_path / "splits", corpus=corpus_path, out=out)

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["n"] == 8
    assert result["model"] == "base-model"
    assert result["split"] == "test"


def test_evaluate_command_missing_splits(tmp_path) -> None:
    with pytest.raises(SystemExit):
        train_cli.evaluate("m", splits=tmp_path / "없음", corpus=tmp_path / "law_docs.json")
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_train_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'train_cli' from 'ragkit.cli'`

- [ ] **Step 3: 구현 (`src/ragkit/cli/train_cli.py`)**

```python
"""DS용 학습·평가 CLI: ragkit split | train | evaluate.

순서: ragkit split → ragkit train --config experiments/exp_00X/config.yaml → ragkit evaluate --model <폴더>
"""

import json
from pathlib import Path

from ragkit.config import get_settings
from ragkit.data import filter_questions, load_corpus, load_questions
from ragkit.embeddings import create_embedding_fn
from ragkit.training.split import load_splits, split_by_law, split_summary, write_splits


def _paths(corpus: Path | None, splits: Path | None) -> tuple[Path, Path]:
    data_dir = get_settings().data_dir
    return (
        corpus or data_dir / "processed" / "law_docs.json",
        splits or data_dir / "splits",
    )


def _fail(message: str) -> None:
    print(f"오류: {message}")
    raise SystemExit(1)


def _load_corpus_or_fail(path: Path) -> list[dict]:
    if not path.exists():
        _fail(f"코퍼스가 없습니다: {path}\n  먼저 실행: uv run python scripts/prepare_law_data.py")
    return load_corpus(path)


def split(
    questions: Path,
    *,
    corpus: Path | None = None,
    out: Path | None = None,
    seed: int = 42,
    test_ratio: float = 0.2,
    dev_ratio: float = 0.1,
    strict: bool = False,
) -> None:
    """질문을 법령 단위로 train/dev/test에 나눈다.

    Args:
        questions: 질문 JSONL 파일 또는 폴더 (예: data/questions).
        corpus: 코퍼스 경로. 기본값은 data/processed/law_docs.json.
        out: 출력 폴더. 기본값은 data/splits.
        seed: 법령 섞기 seed.
        test_ratio: 테마(그룹)별 test 질문 비율.
        dev_ratio: 테마(그룹)별 dev 질문 비율.
        strict: 코퍼스에 없는 질문이 하나라도 있으면 중단한다.
    """
    corpus_path, out = _paths(corpus, out)
    if not Path(questions).exists():
        _fail(f"질문 파일이 없습니다: {questions}\n  law-question-gen 스킬로 먼저 만드세요.")
    docs = _load_corpus_or_fail(corpus_path)
    corpus_by_id = {doc["id"]: doc for doc in docs}

    kept, stats = filter_questions(load_questions(questions), corpus_by_id)
    print(
        f"질문 {len(kept)}개 사용 (코퍼스에 없어 뺀 질문 {stats['missing_positive']}개, "
        f"지운 negative {stats['dropped_negatives']}개)"
    )
    if strict and stats["missing_positive"]:
        _fail("--strict: 코퍼스에 없는 질문이 있습니다.")
    if not kept:
        _fail("남은 질문이 없습니다. 질문과 코퍼스가 같은 버전인지 확인하세요.")

    splits = split_by_law(kept, corpus_by_id, test_ratio=test_ratio, dev_ratio=dev_ratio, seed=seed)
    summary = split_summary(splits, corpus_by_id)
    meta = {
        "seed": seed,
        "test_ratio": test_ratio,
        "dev_ratio": dev_ratio,
        "questions": str(questions),
        "corpus": str(corpus_path),
        "filter": stats,
        **summary,
    }
    write_splits(splits, out, meta)
    for name, info in summary.items():
        print(f"  {name}: 질문 {info['questions']}개, 법령 {len(info['laws'])}개")
        if not info["questions"]:
            print(f"  경고: {name}이 비었습니다 (법령이 적은 테마).")
    print(f"완료 → {out}")


def train(
    config: Path,
    *,
    splits: Path | None = None,
    corpus: Path | None = None,
    model: str | None = None,
    max_steps: int | None = None,
    limit: int | None = None,
) -> None:
    """실험 설정(config.yaml)대로 임베딩 모델을 파인튜닝한다. [train] extra 필요.

    Args:
        config: 실험 설정 경로 (예: experiments/exp_002_finetuned/config.yaml).
        splits: ragkit split 출력 폴더. 기본값은 data/splits.
        corpus: 코퍼스 경로. 기본값은 data/processed/law_docs.json.
        model: 설정의 base 모델 대신 쓸 모델 이름 또는 폴더.
        max_steps: 학습 step 수 제한 (강의에서 짧게 돌려 볼 때).
        limit: 학습 질문 수 제한.
    """
    from ragkit.training.train import load_train_config
    from ragkit.training.train import train as run_train

    corpus_path, splits_dir = _paths(corpus, splits)
    train_config = load_train_config(config)
    overrides = {"model": model, "max_steps": max_steps, "limit": limit}
    train_config = train_config.model_copy(update={k: v for k, v in overrides.items() if v is not None})

    data = load_splits(splits_dir)
    if not data["train"]:
        _fail(f"train 분할이 없습니다: {splits_dir}\n  먼저 실행: ragkit split <질문 파일|폴더>")
    docs = _load_corpus_or_fail(corpus_path)

    settings = get_settings()
    meta = run_train(
        train_config,
        data["train"],
        data["dev"],
        docs,
        query_prefix=settings.query_prefix,
        passage_prefix=settings.passage_prefix,
    )
    print(
        f"완료: {meta['train_examples']}개 예시, {meta['seconds']:.0f}초, "
        f"학습 파라미터 {meta['trainable_params']:,} / {meta['total_params']:,} → {train_config.output_dir}"
    )


def evaluate(
    model: str,
    *,
    split: str = "test",
    splits: Path | None = None,
    corpus: Path | None = None,
    backend: str = "torch",
    out: Path | None = None,
) -> None:
    """모델의 검색 성능을 전체 코퍼스 대상으로 잰다.

    Args:
        model: 모델 이름(base) 또는 학습한 모델 폴더.
        split: 평가할 분할 (test | dev | train).
        splits: ragkit split 출력 폴더. 기본값은 data/splits.
        corpus: 코퍼스 경로. 기본값은 data/processed/law_docs.json.
        backend: 임베딩 백엔드 (torch | onnx). onnx는 <모델 폴더>/onnx/model.onnx가 필요하다.
        out: 결과 JSON 경로. 기본값은 experiments/results/<모델 이름>_<split>.json.
    """
    from ragkit.evaluation import evaluate_retrieval

    corpus_path, splits_dir = _paths(corpus, splits)
    questions = load_splits(splits_dir)[split] if splits_dir.exists() else []
    if not questions:
        _fail(f"{split} 분할이 없거나 비었습니다: {splits_dir}\n  먼저 실행: ragkit split <질문 파일|폴더>")
    docs = _load_corpus_or_fail(corpus_path)

    model_path = Path(model)
    checkpoint = model_path if model_path.is_dir() else None
    embed_fn = create_embedding_fn(model, checkpoint_path=checkpoint, backend=backend)
    settings = get_settings()
    result = evaluate_retrieval(
        embed_fn,
        docs,
        questions,
        query_prefix=settings.query_prefix,
        passage_prefix=settings.passage_prefix,
    )
    result = {"model": model, "split": split, "backend": backend, **result}

    out = out or get_settings().results_dir / f"{model_path.name}_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{model} · {split} ({result['n']}개 질문)")
    for judge in ("doc", "article"):
        metrics = "  ".join(f"{k} {v:.3f}" for k, v in result[judge].items())
        print(f"  [{judge}] {metrics}")
    print(f"결과 → {out}")
```

- [ ] **Step 4: `cli_tool.py`에 등록**

`src/ragkit/cli/cli_tool.py`의 import 블록에 추가:

```python
from ragkit.cli import train_cli
```

`app = cyclopts.App(name="ragkit", help="ragkit: 임베딩 검색 기반 RAG CLI")` 바로 아래에 추가:

```python
# DS용 학습·평가 명령
for _command in (train_cli.split, train_cli.train, train_cli.evaluate):
    app.command(_command)
```

- [ ] **Step 5: 실험 설정 작성**

`experiments/exp_002_finetuned/config.yaml`:

```yaml
# 실험 002: 전체 학습 (모든 파라미터를 학습)
# 실행: ragkit train --config experiments/exp_002_finetuned/config.yaml
#       ragkit evaluate --model models/finetuned/exp_002
name: finetuned
training:
  model: intfloat/multilingual-e5-small
  output_dir: models/finetuned/exp_002
  batch_size: 32
  epochs: 1
  lr: 2.0e-5
  warmup_ratio: 0.1
  max_seq_length: 512
  num_negatives: 1
  seed: 42
```

`experiments/exp_004_lora/config.yaml`:

```yaml
# 실험 004: LoRA (attention query/key/value에 저랭크 행렬만 학습, 끝나면 원래 가중치에 합쳐 저장)
# 실행: ragkit train --config experiments/exp_004_lora/config.yaml
#       ragkit evaluate --model models/finetuned/exp_004
name: lora
training:
  model: intfloat/multilingual-e5-small
  output_dir: models/finetuned/exp_004
  batch_size: 32
  epochs: 1
  lr: 2.0e-4  # LoRA는 학습할 파라미터가 적어 전체 학습보다 큰 학습률을 쓴다
  warmup_ratio: 0.1
  max_seq_length: 512
  num_negatives: 1
  seed: 42
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: [query, key, value]
    save_adapter: true
```

- [ ] **Step 6: 통과 확인**

Run: `uv run pytest tests/test_train_cli.py -v && uv run ragkit --help && uv run ruff check src/ragkit/cli tests/test_train_cli.py`
Expected: 4 passed, 도움말에 `split`, `train`, `evaluate`가 보임, `All checks passed!`

- [ ] **Step 7: 전체 테스트**

Run: `uv run pytest -q`
Expected: 모두 통과 (smoke는 skip)

- [ ] **Step 8: 커밋**

```bash
git add src/ragkit/cli/train_cli.py src/ragkit/cli/cli_tool.py experiments/exp_002_finetuned experiments/exp_004_lora tests/test_train_cli.py
git commit -m "feat(cli): ragkit split/train/evaluate + 전체 학습·LoRA 실험 설정

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 실제 데이터로 전체 흐름 검증 (커밋 없음, 결과 보고)

**Files:** 없음 (출력은 scratchpad `$SP`)

`SP=/private/tmp/claude-501/-Users-jeanboy-workspace-embedding-model-finetuning-tutorial/03bd0337-c62b-485d-aa3b-c1eab0d6376c/scratchpad`
`MAIN=/Users/jeanboy/workspace/embedding-model-finetuning-tutorial`

- [ ] **Step 1: 권장 질문 259개 추출**

`$SP/questions.jsonl`(feat/law-question-gen `22e17b9`의 `data/questions_test/questions.jsonl`, 이미 추출됨)에서 `source.iteration == 2 and source.config == "with_skill"`만 `$SP/best.jsonl`로 저장.

Run: `uv run python -c "import json; rows=[json.loads(l) for l in open('$SP/questions.jsonl')]; best=[r for r in rows if r['source']['iteration']==2 and r['source']['config']=='with_skill']; open('$SP/best.jsonl','w').write('\n'.join(json.dumps(r,ensure_ascii=False) for r in best)); print(len(best))"`
Expected: `259`

- [ ] **Step 2: 분할**

Run: `uv run ragkit split $SP/best.jsonl --corpus $MAIN/data/processed/law_docs.json --out $SP/splits`
Expected: 질문 259개 사용, train/dev/test에 법령이 1개씩(법령 3개가 한 그룹)

- [ ] **Step 3: base 평가**

Run: `uv run ragkit evaluate $MAIN/models/multilingual-e5-small --splits $SP/splits --corpus $MAIN/data/processed/law_docs.json --out $SP/results/base_test.json`
Expected: 지표 출력. 코퍼스 25,806개 임베딩에 수 분 걸릴 수 있다.

- [ ] **Step 4: 전체 학습 / LoRA 짧게 학습 (설정 파일 복사 후 output_dir만 바꿈)**

`experiments/exp_002_finetuned/config.yaml`, `experiments/exp_004_lora/config.yaml`을 `$SP/exp_002.yaml`, `$SP/exp_004.yaml`로 복사하고 `output_dir`을 `$SP/models/exp_002`, `$SP/models/exp_004`로 바꾼다.

Run: `uv run ragkit train --config $SP/exp_002.yaml --splits $SP/splits --corpus $MAIN/data/processed/law_docs.json --model $MAIN/models/multilingual-e5-small`
Run: `uv run ragkit train --config $SP/exp_004.yaml --splits $SP/splits --corpus $MAIN/data/processed/law_docs.json --model $MAIN/models/multilingual-e5-small`
Expected: 각각 `완료: ...개 예시, ...초, 학습 파라미터 ...` (LoRA는 학습 파라미터가 1% 안팎)

- [ ] **Step 5: 학습한 두 모델 평가**

Run: `uv run ragkit evaluate $SP/models/exp_002 --splits $SP/splits --corpus $MAIN/data/processed/law_docs.json --out $SP/results/exp_002_test.json`
Run: `uv run ragkit evaluate $SP/models/exp_004 --splits $SP/splits --corpus $MAIN/data/processed/law_docs.json --out $SP/results/exp_004_test.json`
Expected: 지표 출력

- [ ] **Step 6: 결과 표 보고**

base / 전체 학습 / LoRA의 test `doc`·`article` Recall@1/5/10, MRR@10, 학습 시간, 학습 파라미터 수를 표로 사용자에게 보고한다.
질문이 법령 3개뿐이라(test = 법령 1개) 수치는 참고용이고, 흐름이 끝까지 도는지가 검증 대상이라고 함께 적는다.
