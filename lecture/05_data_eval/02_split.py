"""
실습 1-2 (5교시): 분할 — 질문도 법령도 아니라 조로 나눈다
==========================================================

학습 목표:
- 조 단위 분할(data/splits_article/)의 train · dev · test 질문 수 · 조 수 · 법령 수를 split_meta.json에서 읽는다
- 세 분할의 조가 겹치지 않고, 54개 법령이 모든 분할에 들어갔는지 확인한다
  (test 점수 = "학습한 법령 안의, 학습 질문이 없던 조에서의 성능")
- 질문 단위로 무작위로 나누면 같은 조의 다른 질문이 train과 test에 함께 들어가
  점수가 부풀려진다는 것을 숫자로 본다
- train hard negative에서 dev · test 정답 조의 문서를 뺀 이유를 본다
- (--run) `ragkit split --unit article`을 임시 폴더로 직접 실행해 같은 분할이 나오는지 비교한다

사전 준비:
    uv run python scripts/data_version.py pull v1      # data/questions/ · 코퍼스 (강사 Drive)
    분할이 없으면 이 스크립트가 data/questions/에서 바로 만든다 (몇 초, seed 42라 누가 만들어도 같다)

실행:
    uv run python lecture/05_data_eval/02_split.py           # 분할 확인 (몇 초)
    uv run python lecture/05_data_eval/02_split.py --run     # ragkit split을 임시 폴더에 실행해 비교 (몇 초)
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import cyclopts
from cyclopts import Parameter
from pydantic import BaseModel, Field

from ragkit.config import get_settings
from ragkit.data import load_corpus, relevance_key
from ragkit.training import load_splits

ROOT = Path(__file__).resolve().parents[2]
settings = get_settings()
CORPUS = settings.data_dir / "processed" / "law_docs.json"
QUESTIONS = settings.data_dir / "questions"
SPLITS = settings.data_dir / "splits_article"
META = SPLITS / "split_meta.json"
SEED = 42
# 실험 001과 같은 분할 (experiments/001_baseline/README.md)
SPLIT_ARGS = ["--unit", "article", "--test-ratio", "0.15", "--dev-ratio", "0.1", "--seed", str(SEED)]  # fmt: skip
SPLIT_CMD = f"uv run ragkit split data/questions {' '.join(SPLIT_ARGS)} --out data/splits_article"
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


def split_into(out: Path | str) -> bool:
    """ragkit split --unit article을 돌려 out에 분할을 만든다."""
    sys.stdout.flush()  # 아래 명령의 출력과 순서가 섞이지 않게
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    cmd = [*ragkit_cmd(), "split", str(QUESTIONS), *SPLIT_ARGS, "--out", str(out)]
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode == 0


@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description="ragkit split을 임시 폴더에 실행해 비교")


def cli(*, options: Options | None = None) -> Options:
    """실습 1-2: 조 단위 분할"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)

if not CORPUS.exists() or not any(QUESTIONS.glob("*.jsonl")):
    print(f"코퍼스 또는 질문이 없습니다: {rel(CORPUS)}, {rel(QUESTIONS)}/")
    print("받기: uv run python scripts/data_version.py pull v1")
    sys.exit(1)
if not META.exists():
    print(f"분할이 없어 만든다: {rel(SPLITS)}/")
    print(f"$ {SPLIT_CMD}")
    if not split_into(SPLITS):
        print("ragkit split이 실패했습니다. 위 메시지를 확인하세요.")
        sys.exit(1)

meta = json.loads(META.read_text(encoding="utf-8"))
corpus_by_id = {d["id"]: d for d in load_corpus(CORPUS)}
splits = load_splits(SPLITS)


# ============================================================
# 1. 분할
# ============================================================
section("1. 분할 (split_meta.json)")
print(
    f"질문 {rel(Path(meta['questions']))}, 단위 {meta['unit']}, seed {meta['seed']}, "
    f"test {meta['test_ratio']:.0%} · dev {meta['dev_ratio']:.0%} 목표 (법령마다 조를 나눠서)"
)
total = sum(meta[n]["questions"] for n in NAMES)
print(
    f"\n  {'split':6s} {rjust('질문', 7)} {rjust('비율', 6)} {rjust('조', 6)} {rjust('법령', 5)}  용도"
)
for name in NAMES:
    n = meta[name]["questions"]
    print(
        f"  {name:6s} {n:>7,} {n / total:>6.0%} {meta[name]['articles']:>6,} {len(meta[name]['laws']):>5}  {USE[name]}"
    )
print(f"  합계   {total:>7,}")


# ============================================================
# 2. 조가 겹치지 않는가
# ============================================================
section("2. 확인: 세 분할의 조가 겹치지 않는가")


def article_set(qs: list[dict]) -> set[str]:
    return {relevance_key(corpus_by_id[q["positive_id"]]) for q in qs}


arts = {name: article_set(splits[name]) for name in NAMES}
for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
    both = arts[a] & arts[b]
    print(f"  {a} ∩ {b}: 조 {len(both)}개 {'✓' if not both else '✗'}")
laws = {name: set(meta[name]["laws"]) for name in NAMES}
every = laws["train"] & laws["dev"] & laws["test"]
print(
    f"  세 분할에 모두 든 법령: {len(every)}개 (전체 {len(laws['train'] | laws['dev'] | laws['test'])}개)"
)
print(
    "  · 같은 조의 항 · 호 조각은 한쪽으로 묶는다. 조각마다 같은 조 제목이 붙어 있어 갈라 놓으면 새어 나간다"
)

held = meta.get("heldout_negatives")
if held:
    print(
        f"\ntrain hard negative {held['train_negatives']:,}개 중 dev · test 정답 조의 문서 "
        f"{held['dropped_negatives']:,}개({held['dropped_negatives'] / held['train_negatives']:.1%})를 뺐다"
    )
    print("  · 그대로 두면 test 정답 조가 학습 때 '오답'으로 나온다")
    print(
        f"  · 그래서 오답이 하나도 남지 않은 train 질문 {held['no_negative_questions']:,}개는 학습 예시에서 빠진다"
    )


# ============================================================
# 3. 왜 조 단위인가
# ============================================================
section("3. 왜 질문도 법령도 아니라 조로 나누나")


def leak(train_qs: list[dict], test_qs: list[dict]) -> float:
    """test 질문 중 정답 조문(조 단위)이 train 질문의 정답으로도 나온 비율."""
    seen = {relevance_key(corpus_by_id[q["positive_id"]]) for q in train_qs}
    hit = sum(relevance_key(corpus_by_id[q["positive_id"]]) in seen for q in test_qs)
    return hit / max(len(test_qs), 1)


article_leak = leak(splits["train"], splits["test"])
# 비교용: 같은 질문을 질문 단위로 무작위로 섞어 같은 크기로 나눈 경우
pool = splits["train"] + splits["dev"] + splits["test"]
random.Random(meta["seed"]).shuffle(pool)
n_test = len(splits["test"])
random_test, random_train = pool[:n_test], pool[n_test + len(splits["dev"]) :]
random_leak = leak(random_train, random_test)

print("test 질문 중 '정답 조가 train에서도 정답으로 나온' 비율")
print(f"  조 단위 분할 (이 분할)     {article_leak:6.1%}")
print(f"  질문 단위 무작위 분할      {random_leak:6.1%}")
print(
    "→ 질문 단위로 섞으면 같은 조의 다른 질문을 학습 때 이미 본다. 모델이 그 조를 외워"
)
print("  점수가 부풀려진다")
print("\n법령 단위로 나누면 점수는 부풀지 않지만, test 법령은 학습 때 한 조도 못 본다")
print(
    "  · 운영은 '매뉴얼 전체로 학습하고, 추가되면 다시 학습'이라 처음 보는 법령을 잴 일이 없다"
)
print(
    "  · 조 단위: 모든 법령의 말은 배우고, 아직 아무도 묻지 않은 조로 잰다 = 현업 시나리오"
)


# ============================================================
# 4. (--run) ragkit split 직접 실행
# ============================================================
if args.run:
    section("4. --run: ragkit split을 임시 폴더에 실행")
    with tempfile.TemporaryDirectory() as tmp:
        print(f"$ {SPLIT_CMD.replace('data/splits_article', '<임시 폴더>')}")
        if not split_into(tmp):
            print("ragkit split이 실패했습니다. 위 메시지를 확인하세요.")
        else:
            mine = json.loads(
                (Path(tmp) / "split_meta.json").read_text(encoding="utf-8")
            )
            print("\n지금 분할과 비교")
            for name in NAMES:
                same_file = (Path(tmp) / f"{name}.jsonl").read_bytes() == (
                    SPLITS / f"{name}.jsonl"
                ).read_bytes()
                print(
                    f"  {name:6s} 질문 {mine[name]['questions']:>6,} vs {meta[name]['questions']:>6,}"
                    f"  조 {mine[name]['articles']:>5,} vs {meta[name]['articles']:>5,}"
                    f"  파일 같음 {'✓' if same_file else '✗'}"
                )
            print(
                "→ 같은 질문 · 같은 코퍼스 · 같은 seed면 같은 분할이 나온다 (재현 가능)"
            )
            print("  내가 만든 질문을 더했다면 --out 을 다른 폴더로 주고 비교한다")
    print("임시 폴더는 지웠다. data/splits_article/는 바뀌지 않았다.")

print("""
정리:
- train · dev · test는 조가 겹치지 않고, 54개 법령이 모두 세 분할에 들어간다
- test는 실험마다 마지막에 한 번만 쓴다 (5 · 6교시)
- 질문 단위로 나누면 test 정답 조를 train에서 이미 본다 → 점수가 부풀려진다
- 법령 단위로 나누면 test 법령을 못 배운다 → 현업 시나리오와 다르다
- 다음: 03_metrics.py — 이 test 질문으로 학습 전 모델을 잰다
""")
