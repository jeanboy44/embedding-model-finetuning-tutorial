"""
실습 2-1 (6교시): LoRA 실험 004 — 설정과 파일럿
==============================================

학습 목표:
- 기준 실험 002와 LoRA 실험 004의 설정 파일에서 무엇이 다른지 본다 (lora 블록, 학습률)
- LoRA가 학습하는 파라미터가 왜 0.4%인지 숫자로 확인한다
- 받은 모델 네 개(002 · 004 · 006 · r001_A)의 학습 기록을 나란히 본다
- 파일럿: LoRA 설정으로 몇 step만 직접 돌려 손실이 내려가는지, step당 시간으로 전체 학습이 몇 분일지 본다

사전 준비 (없으면 스크립트가 받는 명령을 알려 주고 끝난다):
    uv run python scripts/data_version.py pull v1     # 코퍼스 + 분할
    models/multilingual-e5-small                                   # scripts/download_model_hf.py
    uv run python scripts/finetuned_drive.py download --all        # models/finetuned/{exp_002,exp_004,exp_006,r001_A}
    (--run) uv sync --extra train                      # sentence-transformers, peft

실행:
    uv run python lecture/06_experiment/01_lora_train.py          # 설정 차이 + 받은 모델의 학습 기록 (몇 초)
    uv run python lecture/06_experiment/01_lora_train.py --run    # 실험 004 설정으로 30 step 파일럿 (약 1~1.5분)

--run은 실험 004 설정 그대로 --output-dir만 임시 폴더로 줘서 돌린다. 받은 모델은 덮어쓰지 않는다.
MNRL · in-batch negative · NO_DUPLICATES 원리는 5교시 lecture/05_data_eval/05_baseline_train.py에서 봤다.
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
from pathlib import Path

import yaml

from ragkit.data import filter_questions, load_corpus
from ragkit.embeddings import create_embedding_fn
from ragkit.training import batch_margins, in_batch_scores, load_splits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
SPLITS = DATA / "splits"
FINETUNED = ROOT / "models" / "finetuned"
BASELINE = ROOT / "experiments" / "exp_002_finetuned" / "config.yaml"
CONFIG = ROOT / "experiments" / "exp_004_lora" / "config.yaml"
MODEL = "intfloat/multilingual-e5-small"
RUNS = {  # 받은 모델 → 무엇을 바꾼 실험인가 (docs/PLAN.md, runs/001_hard_negatives)
    "exp_002": "전체 학습 · 배치 32 · LLM이 고른 오답 1개 (기준)",
    "exp_004": "002 + LoRA (r16, query/key/value) · lr 2e-4",
    "exp_006": "002 + 배치 128",
    "r001_A": "다시 캔 오답 3개 · lr 3e-5 · 4 epoch (에이전트가 돈 실험 001)",
}
BATCH = 4  # 학습 전후를 비교할 배치 크기 (실제 학습은 32)
RUN_STEPS = 30  # --run 학습 step 수
RUN_LIMIT = 960  # --run 학습 질문 수 (배치 32 × 30 step)
LAYERS, HIDDEN = 12, 384  # e5-small: 층 12개, 숨은 차원 384

GET_DATA = "uv run python scripts/data_version.py pull v1"
GET_FINETUNED = "uv run python scripts/finetuned_drive.py download --all"


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
        "준비 상태는 uv run python lecture/03_setup/01_doctor.py 로 한 번에 볼 수 있습니다."
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


sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
parser = argparse.ArgumentParser(description="실습 2-1: LoRA 실험 004, 설정과 파일럿")
parser.add_argument(
    "--run", action="store_true", help=f"실험 004 설정으로 {RUN_STEPS} step 파일럿"
)
args = parser.parse_args()

require(
    [
        (CORPUS, GET_DATA),
        (SPLITS / "train.jsonl", GET_DATA),
        *[(FINETUNED / name / "train_meta.json", GET_FINETUNED) for name in RUNS],
    ]
)
metas = {
    name: json.loads((FINETUNED / name / "train_meta.json").read_text(encoding="utf-8"))
    for name in RUNS
}


# ============================================================
# 1. 설정: 002와 다른 것
# ============================================================
section(f"1. 설정 파일 ({rel(CONFIG)}): 기준 002와 다른 것")
base = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))["training"]
config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
lora = config["training"]
print(f"  {'항목':16s} {'002 기준':24s} 004 LoRA")


def show(value) -> str:
    """설정 값을 장표와 같은 모양으로 (2e-05 → 2e-5, lora 블록 → 한 줄)."""
    if isinstance(value, dict):
        return (
            f"r {value['r']} · alpha {value['alpha']} · dropout {value['dropout']},"
            f" {' · '.join(value['target_modules'])}"
        )
    if isinstance(value, float):
        return f"{value:.0e}".replace("e-0", "e-")
    return str(value)


for key in sorted(set(base) | set(lora)):
    if key in ("output_dir",) or base.get(key) == lora.get(key):
        continue
    before = show(base[key]) if key in base else "없음 (전체 학습)"
    print(f"  {key:16s} {before:24s} {show(lora.get(key))}")
same = [k for k in base if k != "output_dir" and base[k] == lora.get(k)]
print(f"  나머지 {len(same)}개({', '.join(same)})는 002와 같다")
print("  · LoRA는 학습할 파라미터가 적어 보통 전체 학습보다 큰 학습률을 쓴다")
print(
    "  · 엄밀히는 두 가지(LoRA, 학습률)를 같이 바꿨다. 'LoRA + 맞춘 학습률' 한 묶음을 002와 비교한다"
)


# ============================================================
# 2. LoRA가 학습하는 파라미터
# ============================================================
section("2. LoRA가 학습하는 파라미터")
r = lora["lora"]["r"]
targets = lora["lora"]["target_modules"]
expected = LAYERS * len(targets) * r * (HIDDEN + HIDDEN)
meta = metas["exp_004"]
print(
    f"  층 {LAYERS}개 × {'·'.join(targets)} {len(targets)}곳 × r {r} × ({HIDDEN} + {HIDDEN}) = {expected:,}"
)
print(
    f"  받은 모델의 기록: {meta['trainable_params']:,} / {meta['total_params']:,}"
    f" = {meta['trainable_params'] / meta['total_params']:.1%}"
)
print("  · 원래 가중치 W(384×384)는 얼려 두고, 옆에 B(384×r)와 A(r×384)만 학습한다")
print("  · 학습이 끝나면 B×A를 W에 합쳐 보통 모델과 같은 모양으로 저장한다")


# ============================================================
# 3. 받은 모델 네 개의 학습 기록
# ============================================================
section("3. 받은 모델의 학습 기록")
print(
    "모델       학습 파라미터 (비율)     학습 시간  best epoch  epoch별 dev R@5 (학습 중 간이 평가)"
)
for name, what in RUNS.items():
    m = metas[name]
    ratio = m["trainable_params"] / m["total_params"]
    history = " / ".join(f"{h['dev_cosine_recall@5']:.3f}" for h in m["dev_history"])
    before = (m.get("dev_before") or {}).get("dev_cosine_recall@5")
    start = f"학습 전 {before:.3f} → " if before else ""
    print(
        f"{name:10s} {m['trainable_params']:>12,} ({ratio:6.1%})  {m['seconds'] / 60:5.0f}분"
        f"      {m['best_epoch']}      {start}{history}"
    )
    print(f"{'':10s} └ {what}")
print("  · test 성능은 02_compare_runs.py에서 같은 기준으로 잰다")
print("  · r001_A의 dev는 복수 정답 판정을 붙인 값이라 다른 행과 기준이 조금 다르다")


# ============================================================
# 4. 파일럿
# ============================================================
if not args.run:
    section("4. 파일럿을 직접 돌려 보려면")
    print(
        f"  uv run python lecture/06_experiment/01_lora_train.py --run   # {RUN_STEPS} step만 (약 1~1.5분)"
    )
    print("다음: 02_compare_runs.py — 받은 모델들을 test로 비교하고 가설을 확인한다")
    raise SystemExit(0)

section(f"4. --run: 실험 004 설정으로 {RUN_STEPS} step 파일럿")
if (
    importlib.util.find_spec("sentence_transformers") is None
    or importlib.util.find_spec("peft") is None
):
    print("학습 도구가 없습니다. 먼저 실행: uv sync --extra train")
    raise SystemExit(1)

docs = load_corpus(CORPUS)
by_id = {d["id"]: d for d in docs}
train, _ = filter_questions(load_splits(SPLITS)["train"], by_id)
batch: list[dict] = []
for q in train[:RUN_LIMIT]:  # 파일럿이 학습하는 질문 안에서, 법령이 서로 다르게 고른다
    laws = {by_id[b["positive_id"]]["category"] for b in batch}
    if (
        q["hard_negative_ids"]
        and q.get("query_type") != "keyword"
        and by_id[q["positive_id"]]["category"] not in laws
    ):
        batch.append(q)
    if len(batch) == BATCH:
        break
before_scores, before_loss = in_batch_scores(create_embedding_fn(MODEL), batch, by_id)

tmp = Path(tempfile.mkdtemp(prefix="lecture-06-lora-"))
print(f"설정: {rel(CONFIG)} 그대로")
print(
    f"  --output-dir {tmp / 'model'}: 받은 모델({lora['output_dir']}) 대신 임시 폴더에 저장"
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
pilot = json.loads((tmp / "model" / "train_meta.json").read_text(encoding="utf-8"))
after_scores, after_loss = in_batch_scores(
    create_embedding_fn(
        str(tmp / "model"), checkpoint_path=tmp / "model", backend="torch"
    ),
    batch,
    by_id,
)

print("\n확인 1 · 손실이 내려가나")
print(
    f"  같은 배치({BATCH}개)의 MNRL 손실 {before_loss:.3f} → {after_loss:.3f}"
    f" (학습률을 10배로 올렸어도 튀지 않고 내려가면 방향은 맞다)"
)
margins_before, margins_after = (
    batch_margins(before_scores),
    batch_margins(after_scores),
)
improved = int((margins_after > margins_before).sum())
print(f"  {improved}/{len(batch)}개 질문에서 정답과 가장 높은 오답의 거리가 벌어졌다")

print("\n확인 2 · step당 몇 초인가")
estimate = pilot["seconds"] / RUN_STEPS * len(train) / 32 * 3 / 60
print(
    f"  {RUN_STEPS} step에 {pilot['seconds']:.0f}초 (step당 {pilot['seconds'] / RUN_STEPS:.1f}초,"
    f" 모델 로드 포함 전체 {seconds:.0f}초)"
)
print(
    f"  이 속도면 train 전체 3 epoch 약 {estimate:.0f}분."
    f" 받은 모델의 실제 학습은 {meta['seconds'] / 60:.0f}분 (epoch마다 dev 평가와 저장이 더해진다)"
)
shutil.rmtree(tmp, ignore_errors=True)
print(f"\n임시 폴더 {tmp}는 지웠다. 받은 모델은 그대로다.")
print("다음: 02_compare_runs.py — 받은 모델들을 test로 비교하고 가설을 확인한다")
