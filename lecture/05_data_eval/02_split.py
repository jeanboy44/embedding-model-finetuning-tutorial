"""
실습 1-3 (4교시): 분할 — 질문이 아니라 법령으로 나눈다
======================================================

학습 목표:
- 받은 분할(data/splits/)의 train · dev · test 질문 수와 법령 수를 split_meta.json에서 읽는다
- 세 분할의 법령이 겹치지 않는지 확인한다 (test 점수 = "처음 보는 법령에서의 성능")
- 질문 단위로 무작위로 나누면 같은 조문의 다른 질문이 train과 test에 함께 들어가
  점수가 부풀려진다는 것을 숫자로 본다
- (--run) `ragkit split`을 임시 폴더로 직접 실행해 받은 분할과 같은지 비교한다

사전 준비:
    uv run python scripts/data_version.py pull v1      # data/questions/ · data/splits/ · 코퍼스 (강사 Drive)

실행:
    uv run python lecture/04_data/03_split.py           # 받은 분할 확인 (몇 초)
    uv run python lecture/04_data/03_split.py --run     # ragkit split을 임시 폴더에 실행해 비교 (몇 초)
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

from ragkit.config import get_settings
from ragkit.data import load_corpus, relevance_key
from ragkit.training import load_splits

ROOT = Path(__file__).resolve().parents[2]
settings = get_settings()
CORPUS = settings.data_dir / "processed" / "law_docs.json"
QUESTIONS = settings.data_dir / "questions"
SPLITS = settings.data_dir / "splits"
META = SPLITS / "split_meta.json"
NAMES = ("train", "dev", "test")
USE = {"train": "학습", "dev": "epoch · 설정 고르기", "test": "최종 비교에 한 번만"}


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def rjust(text: str, n: int) -> str:
    """한글을 두 칸으로 세어 오른쪽 정렬 (표 머리글용)."""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return " " * max(n - w, 0) + text


def ragkit_cmd() -> list[str]:
    """같은 가상환경의 ragkit CLI (없으면 uv run ragkit)."""
    exe = Path(sys.executable).with_name("ragkit")
    return [str(exe)] if exe.exists() else ["uv", "run", "ragkit"]


parser = argparse.ArgumentParser(description="실습 1-3: 법령 단위 분할")
parser.add_argument(
    "--run", action="store_true", help="ragkit split을 임시 폴더에 실행해 비교"
)
args = parser.parse_args()

if not META.exists() or not CORPUS.exists():
    print(f"분할 또는 코퍼스가 없습니다: {rel(META)}, {rel(CORPUS)}")
    print("받기: uv run python scripts/data_version.py pull v1")
    print("직접 만들기: uv run ragkit split data/questions")
    sys.exit(1)

meta = json.loads(META.read_text(encoding="utf-8"))
corpus_by_id = {d["id"]: d for d in load_corpus(CORPUS)}
splits = load_splits(SPLITS)


# ============================================================
# 1. 받은 분할
# ============================================================
section("1. 받은 분할 (split_meta.json)")
print(
    f"질문 {meta['questions']}, seed {meta['seed']}, "
    f"test {meta['test_ratio']:.0%} · dev {meta['dev_ratio']:.0%} 목표 (테마별로)"
)
total = sum(meta[n]["questions"] for n in NAMES)
print(
    f"\n  {'split':6s} {rjust('질문', 7)} {rjust('비율', 6)} {rjust('법령', 5)}  용도"
)
for name in NAMES:
    n = meta[name]["questions"]
    print(
        f"  {name:6s} {n:>7,} {n / total:>6.0%} {len(meta[name]['laws']):>5}  {USE[name]}"
    )
print(f"  합계   {total:>7,}")

print("\ntest 법령 (학습 때 한 번도 보지 않는다):")
print("  " + ", ".join(meta["test"]["laws"]))


# ============================================================
# 2. 법령이 겹치지 않는가
# ============================================================
section("2. 확인: 세 분할의 법령이 겹치지 않는가")
law_sets = {name: set(meta[name]["laws"]) for name in NAMES}
for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
    both = law_sets[a] & law_sets[b]
    print(
        f"  {a} ∩ {b}: {len(both)}개 {'✓' if not both else '✗ ' + ', '.join(sorted(both))}"
    )

print("\n테마별 법령 수 (테마마다 따로 나눠 test에 모든 테마가 들어간다)")
by_theme: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(NAMES, 0))
law_theme = {d["category"]: d["theme"] for d in corpus_by_id.values()}
for name in NAMES:
    for law in meta[name]["laws"]:
        by_theme[law_theme[law]][name] += 1
print("  테마      " + "".join(f"{n:>7s}" for n in NAMES))
for theme, counts in sorted(by_theme.items()):
    print(f"  {theme:10s}" + "".join(f"{counts[n]:>7}" for n in NAMES))


# ============================================================
# 3. 왜 법령 단위인가
# ============================================================
section("3. 왜 질문이 아니라 법령으로 나누나")


def leak(train_qs: list[dict], test_qs: list[dict]) -> float:
    """test 질문 중 정답 조문(조 단위)이 train 질문의 정답으로도 나온 비율."""
    seen = {relevance_key(corpus_by_id[q["positive_id"]]) for q in train_qs}
    hit = sum(relevance_key(corpus_by_id[q["positive_id"]]) in seen for q in test_qs)
    return hit / max(len(test_qs), 1)


law_leak = leak(splits["train"], splits["test"])
# 비교용: 같은 질문을 질문 단위로 무작위로 섞어 같은 크기로 나눈 경우
pool = splits["train"] + splits["dev"] + splits["test"]
random.Random(meta["seed"]).shuffle(pool)
n_test = len(splits["test"])
random_test, random_train = pool[:n_test], pool[n_test + len(splits["dev"]) :]
random_leak = leak(random_train, random_test)

print("test 질문 중 '정답 조문이 train에서도 정답으로 나온' 비율")
print(f"  법령 단위 분할 (받은 것)   {law_leak:6.1%}")
print(f"  질문 단위 무작위 분할      {random_leak:6.1%}")
print(
    "→ 질문 단위로 섞으면 같은 조문의 다른 질문을 학습 때 이미 본다. 모델이 그 조문을 외워"
)
print("  점수가 부풀려진다. 법령 단위로 나눠야 '처음 보는 법령에서의 성능'을 잰다")


# ============================================================
# 4. (--run) ragkit split 직접 실행
# ============================================================
if args.run:
    section("4. --run: ragkit split을 임시 폴더에 실행")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            *ragkit_cmd(),
            "split",
            str(QUESTIONS),
            "--out",
            tmp,
            "--seed",
            str(meta["seed"]),
        ]
        print(
            f"$ uv run ragkit split {rel(QUESTIONS)} --out <임시 폴더> --seed {meta['seed']}"
        )
        sys.stdout.flush()  # 아래 명령의 출력과 순서가 섞이지 않게
        env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
        result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if result.returncode != 0:
            print("ragkit split이 실패했습니다. 위 메시지를 확인하세요.")
        else:
            mine = json.loads(
                (Path(tmp) / "split_meta.json").read_text(encoding="utf-8")
            )
            print("\n받은 분할과 비교")
            for name in NAMES:
                same_file = (Path(tmp) / f"{name}.jsonl").read_bytes() == (
                    SPLITS / f"{name}.jsonl"
                ).read_bytes()
                same_laws = mine[name]["laws"] == meta[name]["laws"]
                print(
                    f"  {name:6s} 질문 {mine[name]['questions']:>6,} vs {meta[name]['questions']:>6,}"
                    f"  법령 같음 {'✓' if same_laws else '✗'}  파일 같음 {'✓' if same_file else '✗'}"
                )
            print(
                "→ 같은 질문 · 같은 코퍼스 · 같은 seed면 같은 분할이 나온다 (재현 가능)"
            )
            print("  내가 만든 질문을 더했다면 --out 을 다른 폴더로 주고 비교한다")
    print("임시 폴더는 지웠다. data/splits/는 바뀌지 않았다.")

print("""
정리:
- train · dev · test는 법령이 겹치지 않는다. test는 실습 3의 최종 비교에 한 번만 쓴다
- 질문 단위로 나누면 test 정답 조문을 train에서 이미 본다 → 점수가 부풀려진다
- 다음: 실습 2 (lecture/05_evaluate/) — 이 test 질문으로 학습 전 모델을 잰다
""")
