"""
실습 3-1 (6교시): 학습 원리와 맛보기
====================================

학습 목표:
- 학습 데이터 한 줄(질문 · 정답 조문 · hard negative)이 어떻게 생겼는지 본다
- MNRL(MultipleNegativesRankingLoss)이 배치 안의 다른 정답을 오답으로 쓰는 방식(in-batch negative)을 숫자로 확인한다
- NO_DUPLICATES가 왜 필요한지, hard negative가 무엇인지, 전체 학습과 LoRA가 어떻게 다른지 짧게 본다
- 학습은 맛보기만 한다. 전체 학습(실험 002 약 40분)은 강사가 미리 한 모델과 학습 기록을 받아 읽는다

사전 준비 (없으면 스크립트가 받는 명령을 알려 주고 끝난다):
    uv run python scripts/data_version.py pull v1     # 코퍼스 + 분할
    models/multilingual-e5-small, models/finetuned/{exp_002,exp_004,exp_006,r001_A}   # 강사 Drive
    (--run) uv sync --extra train                      # sentence-transformers, peft

실행:
    uv run python lecture/03_train/01_train_taste.py          # 원리 + 받은 모델의 학습 기록 (1분 안쪽)
    uv run python lecture/03_train/01_train_taste.py --run    # ragkit train으로 몇 step만 직접 학습 (몇 분)

--run은 실험 002 설정 그대로 --output-dir만 임시 폴더로 줘서 돌린다. 받은 모델은 덮어쓰지 않는다.
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from ragkit.data import doc_text, filter_questions, load_corpus
from ragkit.embeddings import create_embedding_fn, format_passages, format_queries
from ragkit.training import load_splits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
SPLITS = DATA / "splits"
FINETUNED = ROOT / "models" / "finetuned"
CONFIG = ROOT / "experiments" / "exp_002_finetuned" / "config.yaml"
MODEL = "intfloat/multilingual-e5-small"
RUNS = {  # 받은 모델 → 무엇을 바꾼 실험인가 (docs/PLAN.md, runs/001_hard_negatives)
    "exp_002": "전체 학습 · 배치 32 · LLM이 고른 오답 1개 (기준)",
    "exp_004": "002 + LoRA (r16, query/key/value)",
    "exp_006": "002 + 배치 128",
    "r001_A": "다시 캔 오답 3개 · lr 3e-5 · 4 epoch",
}
BATCH = 4  # in-batch negative 예시 배치 크기 (실제 학습은 32)
SCALE = 20.0  # MNRL이 코사인 유사도에 곱하는 값 (sentence-transformers 기본)
RUN_STEPS = 30  # --run 학습 step 수
RUN_LIMIT = 960  # --run 학습 질문 수 (배치 32 × 30 step)

GET_DATA = "uv run python scripts/data_version.py pull v1"
FROM_DRIVE = "강사 Drive에서 받아 {path}에 둔다"


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def require(items: list[tuple[Path, str]]) -> None:
    """필요한 파일이 없으면 무엇이 없고 어떻게 받는지 알려 주고 끝낸다."""
    missing = [(path, how) for path, how in items if not path.exists()]
    if not missing:
        return
    print("필요한 산출물이 없습니다.")
    for path, how in missing:
        print(f"  - {rel(path)}\n      받기: {how.format(path=rel(path))}")
    print(
        "준비 상태는 uv run python lecture/00_setup/01_check_env.py 로 한 번에 볼 수 있습니다."
    )
    raise SystemExit(1)


def ragkit(*args: str) -> float:
    """ragkit CLI를 저장소 루트에서 실행하고 걸린 시간(초)을 돌려준다."""
    shown = [rel(Path(a)) if a.startswith(str(ROOT)) else a for a in args]
    print(f"$ uv run ragkit {' '.join(shown)}")
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONUNBUFFERED": "1"}
    start = time.perf_counter()
    done = subprocess.run(
        [str(Path(sys.executable).parent / "ragkit"), *args],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if done.returncode != 0:
        print(
            f"ragkit {args[0]}이 실패했습니다 (종료 코드 {done.returncode}). 위 메시지를 확인하세요."
        )
        raise SystemExit(done.returncode)
    return time.perf_counter() - start


def in_batch(
    embed, batch: list[dict], by_id: dict[str, dict]
) -> tuple[np.ndarray, float]:
    """질문 × (배치의 정답들 + hard negative들) 점수표와 MNRL 손실(평균)."""
    queries = embed(format_queries([q["query"] for q in batch]))
    passages = [doc_text(by_id[q["positive_id"]]) for q in batch]
    passages += [doc_text(by_id[q["hard_negative_ids"][0]]) for q in batch]
    scores = queries @ embed(format_passages(passages)).T
    logits = SCALE * scores
    logits -= logits.max(axis=1, keepdims=True)
    log_softmax = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    loss = float(-np.mean([log_softmax[i, i] for i in range(len(batch))]))
    return scores, loss


def show_matrix(scores: np.ndarray, loss: float) -> None:
    n = scores.shape[0]
    header = "".join(f"  정답{j + 1} " for j in range(n)) + "".join(
        f"  오답{j + 1} " for j in range(n)
    )
    print(f"         {header}")
    for i, row in enumerate(scores):
        cells = "".join(
            f"  [{v:.2f}]" if i == j else f"   {v:.2f} " for j, v in enumerate(row)
        )
        print(f"  질문{i + 1}  {cells}")
    print(f"  MNRL 손실 {loss:.3f} (대각선 [ ]이 그 줄에서 클수록 작아진다)")


sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
parser = argparse.ArgumentParser(description="실습 3-1: 학습 원리와 맛보기")
parser.add_argument(
    "--run", action="store_true", help=f"ragkit train으로 {RUN_STEPS} step만 직접 학습"
)
args = parser.parse_args()

require(
    [
        (CORPUS, GET_DATA),
        (SPLITS / "train.jsonl", GET_DATA),
        *[(FINETUNED / name / "train_meta.json", FROM_DRIVE) for name in RUNS],
    ]
)
docs = load_corpus(CORPUS)
by_id = {d["id"]: d for d in docs}
train, _ = filter_questions(load_splits(SPLITS)["train"], by_id)


# ============================================================
# 1. 학습 데이터 한 줄
# ============================================================
section("1. 학습 데이터 한 줄: 질문 · 정답 · hard negative")
example = next(
    q for q in train if q.get("query_type") == "situation" and q["hard_negative_ids"]
)
print(f"질문            {example['query']}")
print(f"정답(positive)  {by_id[example['positive_id']]['title']}")
print(f"오답(hard neg.) {by_id[example['hard_negative_ids'][0]]['title']}")
print(
    "  · hard negative = 정답과 비슷해 보이지만 답이 아닌 조문. 질문을 만들 때 LLM이 같은 법에서 골랐다"
)
print(
    '  · e5 접두어("query: " · "passage: ")는 데이터가 아니라 코드(ragkit.embeddings.format_queries)에서 붙인다'
)
print(f"  · train 질문 {len(train):,}개가 이런 줄로 바뀌어 학습에 들어간다")


# ============================================================
# 2. MNRL: 배치 안의 다른 정답이 공짜 오답이 된다
# ============================================================
section(f"2. MNRL과 in-batch negative (배치 {BATCH}개로 축소)")
batch: list[dict] = []
for q in train[
    :RUN_LIMIT
]:  # --run이 학습하는 질문 안에서, 법령이 서로 다르게 고른다 (학습 전후 비교용)
    laws = {by_id[b["positive_id"]]["category"] for b in batch}
    if (
        q["hard_negative_ids"]
        and q.get("query_type") != "keyword"
        and by_id[q["positive_id"]]["category"] not in laws
    ):
        batch.append(q)
    if len(batch) == BATCH:
        break
for i, q in enumerate(batch, 1):
    print(
        f"  질문{i} {q['query'][:34]}  (정답: {by_id[q['positive_id']]['title'][:30]})"
    )
print(
    "\n학습 전 e5-small의 코사인 유사도 (줄 = 질문, 칸 = 배치의 정답 4개 + 각 질문의 hard negative 4개)"
)
base_embed = create_embedding_fn(MODEL)
before_scores, before_loss = in_batch(base_embed, batch, by_id)
show_matrix(before_scores, before_loss)
print("  · 질문1의 정답은 '정답1' 하나, 나머지 7칸은 모두 오답으로 쓴다")
print("  · 실제 학습은 배치 32: 질문 하나에 in-batch 오답 31개 + hard negative")
print("  · 학습은 대각선을 키우고 나머지를 줄이는 방향으로 가중치를 바꾼다")


# ============================================================
# 3. NO_DUPLICATES
# ============================================================
section("3. NO_DUPLICATES: 같은 조문이 한 배치에 두 번 들어가면")
counts = Counter(q["positive_id"] for q in train)
shared = sum(n for n in counts.values() if n > 1)
top_id, top_n = counts.most_common(1)[0]
print(
    f"train 질문 {len(train):,}개 중 정답 조문을 다른 질문과 같이 쓰는 질문 {shared:,}개 ({shared / len(train):.0%})"
)
print(f"  가장 많은 예: {by_id[top_id]['title']} ← 질문 {top_n}개")
print(
    "  두 질문이 한 배치에 들어가면 서로의 정답(= 같은 조문)을 '오답'으로 배운다 → 가짜 오답"
)
print(
    "  그래서 학습은 BatchSamplers.NO_DUPLICATES로 같은 문서가 한 배치에 두 번 들어가지 않게 묶는다"
)


# ============================================================
# 4. 설정 파일과 전체 학습 vs LoRA
# ============================================================
section("4. 설정 파일 (실험 002) 과 받은 모델의 학습 기록")
config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
print(f"{rel(CONFIG)} 의 training:")
for key, value in config["training"].items():
    print(f"  {key}: {value}")

print(
    "\n모델       학습 파라미터 (비율)     학습 시간  best epoch  epoch별 dev R@5 (학습 중 간이 평가)"
)
for name, what in RUNS.items():
    meta = json.loads(
        (FINETUNED / name / "train_meta.json").read_text(encoding="utf-8")
    )
    ratio = meta["trainable_params"] / meta["total_params"]
    history = " / ".join(f"{h['dev_cosine_recall@5']:.3f}" for h in meta["dev_history"])
    before = (meta.get("dev_before") or {}).get("dev_cosine_recall@5")
    start = f"학습 전 {before:.3f} → " if before else ""
    print(
        f"{name:10s} {meta['trainable_params']:>12,} ({ratio:6.1%})  {meta['seconds'] / 60:5.0f}분"
        f"      {meta['best_epoch']}      {start}{history}"
    )
    print(f"{'':10s} └ {what}")
print(
    "  · LoRA는 attention에 작은 행렬만 덧붙여 학습한다. 파라미터는 적지만 성능은 비교표(02_compare_runs.py)에서 본다"
)
print("  · epoch는 dev R@5로 자동으로 고른다(best epoch). test는 보지 않는다")
print("  · r001_A의 dev는 복수 정답 판정을 붙인 값이라 다른 행과 기준이 조금 다르다")


# ============================================================
# 5. 맛보기 학습
# ============================================================
if not args.run:
    section("5. 직접 학습해 보려면")
    print(
        f"  uv run python lecture/03_train/01_train_taste.py --run   # {RUN_STEPS} step만 (몇 분)"
    )
    print(
        "  전체 학습은 just finetune-suite (분할 → 002·004·006 학습 → 비교표, 약 2시간)"
    )
    print("다음: 02_compare_runs.py — 받은 모델들을 test로 비교하고 가설을 확인한다")
    raise SystemExit(0)

section(f"5. --run: ragkit train으로 {RUN_STEPS} step 맛보기")
if (
    importlib.util.find_spec("sentence_transformers") is None
    or importlib.util.find_spec("peft") is None
):
    print("학습 도구가 없습니다. 먼저 실행: uv sync --extra train")
    raise SystemExit(1)
tmp = Path(tempfile.mkdtemp(prefix="lecture-03-1-"))
print(f"설정: {CONFIG.relative_to(ROOT)} 그대로")
print(
    f"  --output-dir {tmp / 'model'}: 받은 모델({config['training']['output_dir']}) 대신 임시 폴더에 저장"
)
print(
    f"  --limit {RUN_LIMIT}: 질문 {RUN_LIMIT}개만 · --max-steps {RUN_STEPS} · --no-dev-eval: dev 평가(코퍼스 임베딩 2회) 생략\n"
)
seconds = ragkit(
    "train",
    str(CONFIG),
    "--output-dir",
    str(tmp / "model"),
    "--max-steps",
    str(RUN_STEPS),
    "--limit",
    str(RUN_LIMIT),
    "--no-dev-eval",
)
meta = json.loads((tmp / "model" / "train_meta.json").read_text(encoding="utf-8"))
print(
    f"\n학습 {meta['seconds']:.0f}초 (step당 {meta['seconds'] / RUN_STEPS:.1f}초, 모델 로드 포함 전체 {seconds:.0f}초)"
)
print(
    f"  같은 속도로 train 전체 3 epoch를 돌리면 약 {meta['seconds'] / RUN_STEPS * len(train) / 32 * 3 / 60:.0f}분"
)

print("\n같은 배치를 맛보기 모델로 다시 재면:")
taste_embed = create_embedding_fn(
    str(tmp / "model"), checkpoint_path=tmp / "model", backend="torch"
)
after_scores, after_loss = in_batch(taste_embed, batch, by_id)
show_matrix(after_scores, after_loss)


def margins(scores: np.ndarray) -> np.ndarray:
    """질문마다 정답 점수 - 가장 높은 오답 점수 (클수록 정답이 확실히 앞선다)."""
    others = scores.copy()
    np.fill_diagonal(others[:, : len(scores)], -np.inf)
    return np.diag(scores) - others.max(axis=1)


before_margin, after_margin = margins(before_scores), margins(after_scores)
print("\n질문별 여유 (정답 점수 - 가장 높은 오답 점수, 0보다 크면 정답이 1위)")
for i, (a, b) in enumerate(zip(before_margin, after_margin), 1):
    print(f"  질문{i}  {a:+.3f} → {b:+.3f}")
improved = int((after_margin > before_margin).sum())
print(
    f"  {improved}/{len(batch)}개 질문에서 여유가 커졌다. 배치 손실 {before_loss:.3f} → {after_loss:.3f}"
)
print(
    f"  점수 범위: 학습 전 {before_scores.min():.2f}~{before_scores.max():.2f}에 몰려 있던 점수가 "
    f"{after_scores.min():.2f}~{after_scores.max():.2f}로 퍼졌다 (MNRL이 정답과 오답을 떼어 놓는다)"
)
print(
    "  30 step 맛보기라 질문마다 한 번씩만 봤다. 전체 학습(3 epoch)의 결과는 02_compare_runs.py에서 본다"
)
shutil.rmtree(tmp, ignore_errors=True)
print(f"\n임시 폴더 {tmp}는 지웠다 (모델 약 470MB). 받은 모델은 그대로다.")
print("다음: 02_compare_runs.py — 받은 모델들을 test로 비교하고 가설을 확인한다")
