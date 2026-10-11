"""
실습 1-4 (5교시): 기본 실험 001 — 학습 원리와 파일럿
=====================================================

학습 목표:
- 학습 데이터 한 줄(질문 · 정답 조문 · hard negative)이 어떻게 생겼는지 본다
- MNRL(MultipleNegativesRankingLoss)이 배치 안의 다른 정답을 오답으로 쓰는 방식(in-batch negative)을 숫자로 확인한다
- NO_DUPLICATES가 왜 필요한지 본다
- 기본 실험 001의 설정 파일과 받은 모델의 학습 기록을 읽는다. 전체 학습(M4 Pro 39분)은 강사가 미리 했다
- 파일럿: 같은 설정으로 몇 step만 직접 돌려 손실 · 메모리 · step당 시간을 확인한다

무엇을 하나 (기본 모드, 1분 안쪽):
    1. 학습 데이터 한 줄: train 질문 하나의 질문 · 정답 · hard negative 제목, 학습에 들어가는 질문 수
    2. MNRL과 in-batch negative: 질문 4개 배치의 코사인 점수표와 손실 (ragkit.training.in_batch_scores)
    3. NO_DUPLICATES: 정답 조문을 다른 질문과 같이 쓰는 train 질문 수 (가짜 오답의 근거)
    4. 설정 파일과 학습 기록: experiments/001_baseline/config.yaml, 받은 모델의 train_meta.json
    5. (--run) `ragkit train`을 30 step만 임시 폴더에 돌리고, 2의 배치를 파일럿 모델로 다시 재서 비교

사전 준비 (없으면 스크립트가 받는 명령을 알려 주고 끝난다):
    uv run python scripts/data_version.py pull v1     # 코퍼스 + 질문
    uv run python lecture/05_data_eval/02_split.py    # 조 단위 분할 data/splits_article/ (없으면 만든다)
    models/multilingual-e5-small                                   # scripts/download_model_hf.py
    uv run python scripts/finetuned_drive.py download 001_baseline     # models/finetuned/001_baseline
    (--run) uv sync --extra train                      # sentence-transformers

실행:
    uv run python lecture/05_data_eval/04_baseline_train.py          # 원리 + 받은 모델의 학습 기록 (1분 안쪽)
    uv run python lecture/05_data_eval/04_baseline_train.py --run    # 실험 001 설정으로 30 step 파일럿 (모델 로드 포함 2~3분)

--run은 실험 001 설정 그대로 --output-dir만 임시 폴더로 줘서 돌린다. 받은 모델은 덮어쓰지 않는다.
6교시에는 학습 없이 이 001 모델을 판정하고 오답을 분석한다 (lecture/06_experiment/).
"""

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

import cyclopts
import numpy as np
import yaml
from cyclopts import Parameter
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ragkit.data import filter_questions, load_corpus
from ragkit.embeddings import create_embedding_fn
from ragkit.training import batch_margins, in_batch_scores, load_splits

sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
console = Console(emoji=False)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
SPLITS = DATA / "splits_article"  # 조 단위 분할 (02_split.py)
CONFIG = ROOT / "experiments" / "001_baseline" / "config.yaml"
RECEIVED = ROOT / "models" / "finetuned" / "001_baseline"
MODEL = "intfloat/multilingual-e5-small"
BATCH = 4  # in-batch negative 예시 배치 크기 (실제 학습은 32)
RUN_STEPS = 30  # --run 학습 step 수
RUN_LIMIT = 960  # --run 학습 질문 수 (배치 32 × 30 step)

GET_DATA = "uv run python scripts/data_version.py pull v1"
GET_SPLITS = "uv run python lecture/05_data_eval/02_split.py"
GET_FINETUNED = "uv run python scripts/finetuned_drive.py download 001_baseline"


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold]{escape(title)}[/]", style="blue")


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
    lines = [
        f"  - {escape(rel(path))}\n      받기: [dim]{escape(how.format(path=rel(path)))}[/]"
        for path, how in missing
    ]
    lines.append(
        "준비 상태는 [dim]uv run python lecture/03_setup/01_doctor.py[/] 로 한 번에 볼 수 있습니다."
    )
    console.print(
        Panel("\n".join(lines), title="필요한 산출물이 없습니다", border_style="red")
    )
    raise SystemExit(1)


def ragkit(*args: str) -> float:
    """ragkit CLI를 저장소 루트에서 실행하고 걸린 시간(초)을 돌려준다."""
    shown = [rel(Path(a)) if a.startswith(str(ROOT)) else a for a in args]
    console.print(f"[dim]$ uv run ragkit {escape(' '.join(shown))}[/]")
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONUNBUFFERED": "1"}
    start = time.perf_counter()
    done = subprocess.run(
        [str(Path(sys.executable).parent / "ragkit"), *args],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if done.returncode != 0:
        console.print(
            f"[red]ragkit {escape(args[0])}이 실패했습니다 (종료 코드 {done.returncode}).[/] 위 메시지를 확인하세요."
        )
        raise SystemExit(done.returncode)
    return time.perf_counter() - start


def show_matrix(scores: np.ndarray, loss: float) -> None:
    n = scores.shape[0]
    table = Table(box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("")
    for j in range(n):
        table.add_column(f"정답{j + 1}", justify="right")
    for j in range(n):
        table.add_column(f"오답{j + 1}", justify="right")
    for i, row in enumerate(scores):
        cells = [
            f"[bold]{escape(f'[{v:.2f}]')}[/]" if i == j else f"{v:.2f}"
            for j, v in enumerate(row)
        ]
        table.add_row(f"질문{i + 1}", *cells)
    console.print(table)
    console.print(
        f"  MNRL 손실 [bold]{loss:.3f}[/] (대각선 {escape('[ ]')}이 그 줄에서 클수록 작아진다)",
        highlight=False,
    )


@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description=f"실험 001 설정으로 {RUN_STEPS} step 파일럿")


def cli(*, options: Options | None = None) -> Options:
    """실습 1-4: 기본 실험 001, 학습 원리와 파일럿"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)

require(
    [
        (CORPUS, GET_DATA),
        (SPLITS / "train.jsonl", GET_SPLITS),
        (RECEIVED / "train_meta.json", GET_FINETUNED),
    ]
)
docs = load_corpus(CORPUS)
by_id = {d["id"]: d for d in docs}
# 코퍼스에 없는 id를 가리키는 질문은 거른다 (학습 코드와 같은 전처리)
train, _ = filter_questions(load_splits(SPLITS)["train"], by_id)


# ============================================================
# 1. 학습 데이터 한 줄
# ============================================================
# 학습에 들어가는 한 줄은 (질문, 정답 조문, hard negative 조문)이다. situation 질문 하나로
# 세 칸의 제목을 찍고, train 9,126개 중 hard negative가 남은 8,689개만 학습 예시가 됨을 보인다
# (02_split.py에서 dev · test 정답 조의 문서를 오답에서 뺐기 때문).
section("1. 학습 데이터 한 줄: 질문 · 정답 · hard negative")
example = next(
    q for q in train if q.get("query_type") == "situation" and q["hard_negative_ids"]
)  # 오답이 남아 있는 situation 질문 중 첫 번째
console.print(f"질문            {escape(example['query'])}", highlight=False)
console.print(
    f"정답(positive)  {escape(by_id[example['positive_id']]['title'])}", highlight=False
)
console.print(
    f"오답(hard neg.) {escape(by_id[example['hard_negative_ids'][0]]['title'])}",
    highlight=False,
)
console.print(
    "  · hard negative = 정답과 비슷해 보이지만 답이 아닌 조문. 질문을 만들 때 LLM이 같은 법에서 골랐다"
)
console.print(
    '  · e5 접두어("query: " · "passage: ")는 데이터가 아니라 코드(ragkit.embeddings.format_queries)에서 붙인다'
)
with_neg = sum(bool(q["hard_negative_ids"]) for q in train)  # = 학습 예시 수 8,689
console.print(
    f"  · train 질문 {len(train):,}개 중 hard negative가 남은 [bold]{with_neg:,}[/]개가 이런 줄로 바뀌어 학습에 들어간다",
    highlight=False,
)
console.print(
    "    (dev · test 정답 조의 문서는 오답에서 뺐다. 그래서 오답이 비어 빠지는 질문이 있다 — 02_split.py)"
)


# ============================================================
# 2. MNRL: 배치 안의 다른 정답이 공짜 오답이 된다
# ============================================================
# 법령이 서로 다른 질문 4개로 작은 배치를 만들고, 학습 전 e5(ragkit.embeddings.create_embedding_fn)로
# ragkit.training.in_batch_scores를 불러 질문 × (정답 4 + hard negative 4) 코사인 점수표와 MNRL 손실을 찍는다.
# 출력에서 볼 것: 점수가 0.78~0.88에 몰려 대각선이 뚜렷하지 않다, 손실 1.514. --run 뒤에 같은 배치를 다시 잰다.
section(f"2. MNRL과 in-batch negative (배치 {BATCH}개로 축소)")
batch: list[dict] = []
for q in train[
    :RUN_LIMIT
]:  # --run이 학습하는 질문 안에서, 법령이 서로 다르게 고른다 (학습 전후 비교용)
    laws = {by_id[b["positive_id"]]["category"] for b in batch}
    if (
        q["hard_negative_ids"]  # hard negative 칸이 있어야 점수표 오른쪽 4칸이 찬다
        and q.get("query_type") != "keyword"  # 쉬운 keyword 질문은 뺀다
        and by_id[q["positive_id"]]["category"] not in laws  # 법령이 서로 다르게
    ):
        batch.append(q)
    if len(batch) == BATCH:
        break
for i, q in enumerate(batch, 1):
    console.print(
        f"  질문{i} {escape(q['query'][:34])}  (정답: {escape(by_id[q['positive_id']]['title'][:30])})",
        highlight=False,
    )
console.print(
    "\n학습 전 e5-small의 코사인 유사도 (줄 = 질문, 칸 = 배치의 정답 4개 + 각 질문의 hard negative 4개)"
)
before_scores, before_loss = in_batch_scores(create_embedding_fn(MODEL), batch, by_id)
show_matrix(before_scores, before_loss)
console.print("  · 질문1의 정답은 '정답1' 하나, 나머지 7칸은 모두 오답으로 쓴다")
console.print(
    "  · 실제 학습은 배치 32: 질문 하나에 in-batch 오답 31개 + hard negative"
)
console.print("  · 학습은 대각선을 키우고 나머지를 줄이는 방향으로 가중치를 바꾼다")


# ============================================================
# 3. NO_DUPLICATES
# ============================================================
# 정답 조문(positive_id)이 같은 train 질문이 얼마나 많은지 센다 (3,466개, 38%).
# 그런 질문 둘이 한 배치에 들면 서로의 정답이 in-batch 오답이 되어 같은 조문을 밀어내는 가짜 오답이 생긴다.
# 그래서 config의 학습은 BatchSamplers.NO_DUPLICATES로 묶는다는 근거다.
section("3. NO_DUPLICATES: 같은 조문이 한 배치에 두 번 들어가면")
counts = Counter(q["positive_id"] for q in train)  # 정답 문서 id → 질문 수
shared = sum(n for n in counts.values() if n > 1)  # 정답을 같이 쓰는 질문 수
top_id, top_n = counts.most_common(1)[0]
console.print(
    f"train 질문 {len(train):,}개 중 정답 조문을 다른 질문과 같이 쓰는 질문 [bold]{shared:,}[/]개 ({shared / len(train):.0%})",
    highlight=False,
)
console.print(
    f"  가장 많은 예: {escape(by_id[top_id]['title'])} ← 질문 {top_n}개", highlight=False
)
console.print(
    "  두 질문이 한 배치에 들어가면 서로의 정답(= 같은 조문)을 '오답'으로 배운다 → 가짜 오답"
)
console.print(
    "  그래서 학습은 BatchSamplers.NO_DUPLICATES로 같은 문서가 한 배치에 두 번 들어가지 않게 묶는다"
)


# ============================================================
# 4. 설정 파일과 받은 모델의 학습 기록
# ============================================================
# experiments/001_baseline/config.yaml의 training 항목을 그대로 찍고, 받은 모델의 train_meta.json에서
# 학습 시간(39분) · 학습 파라미터 수 · best epoch · epoch별 dev R@5(0.540 → 0.697 / 0.710 / 0.717)를 읽는다.
# epoch는 dev로 고르고 test는 보지 않는다. test 결과(0.545 → 0.744)는 실험 001 기록에서 인용한다.
section(f"4. 설정 파일 ({rel(CONFIG)}) 과 받은 모델의 학습 기록")
config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
console.print(f"[dim]{escape(rel(CONFIG))}[/] 의 training:")
for key, value in config["training"].items():
    console.print(f"  {escape(str(key))}: {escape(str(value))}")

meta = json.loads((RECEIVED / "train_meta.json").read_text(encoding="utf-8"))
# dev_history = epoch마다 학습 중 잰 dev R@5, dev_before = 학습 전 (옛 기록에는 없을 수 있다)
history = " / ".join(f"{h['dev_cosine_recall@5']:.3f}" for h in meta["dev_history"])
before = (meta.get("dev_before") or {}).get("dev_cosine_recall@5")
start = f"학습 전 {before:.3f} → " if before else ""
console.print(
    f"\n받은 모델 [dim]{escape(rel(RECEIVED))}[/]: 학습 [bold]{meta['seconds'] / 60:.0f}분[/], "
    f"학습 파라미터 {meta['trainable_params']:,}개 (전부), best epoch {meta['best_epoch']}",
    highlight=False,
)
console.print(f"  epoch별 dev R@5 (학습 중 간이 평가): {start}{history}")
console.print("  · epoch는 dev R@5로 자동으로 고른다(best epoch). test는 보지 않는다")
console.print(
    "  · test는 실험마다 마지막에 한 번만 잰다: 001은 R@5 0.545 → 0.744 (학습에 없던 조 942개, 질문 1,863개)"
)


# ============================================================
# 5. 파일럿
# ============================================================
# 기본 모드: 파일럿과 전체 학습 명령만 안내하고 끝낸다.
# --run: 실험 001 설정 그대로 `ragkit train`을 --output-dir 임시 폴더, --limit 960, --max-steps 30,
# --no-dev-eval로 돌린다 (sentence-transformers 필요, 모델 로드 포함 2~3분). 끝나면 2의 배치를
# 파일럿 모델로 다시 재서 점수표 · 손실 · 질문별 여유를 학습 전과 비교하고 임시 폴더를 지운다.
if not args.run:
    section("5. 파일럿을 직접 돌려 보려면")
    console.print(
        f"  [dim]uv run python lecture/05_data_eval/04_baseline_train.py --run[/]   # {RUN_STEPS} step만 (모델 로드 포함 2~3분)"
    )
    console.print(
        f"  전체 학습은 [dim]uv run ragkit train {escape(rel(CONFIG))} --splits {escape(rel(SPLITS))} --output-dir models/finetuned/my_001[/] (M4 Pro 39분, 받은 모델을 덮어쓰지 않게 출력 폴더를 바꾼다)"
    )
    console.print("다음(6교시): 기본 실험 001을 판정하고 오답을 쪼개 다음 가설을 찾는다")
    raise SystemExit(0)

section(f"5. --run: 실험 001 설정으로 {RUN_STEPS} step 파일럿")
if importlib.util.find_spec("sentence_transformers") is None:  # extra train 전
    console.print(
        Panel(
            "학습 도구가 없습니다. 먼저 실행: [dim]uv sync --extra train[/]",
            title="필요한 산출물이 없습니다",
            border_style="red",
        )
    )
    raise SystemExit(1)
tmp = Path(tempfile.mkdtemp(prefix="lecture-05-baseline-"))
console.print(f"설정: {escape(rel(CONFIG))} 그대로")
console.print(
    f"  --output-dir {escape(str(tmp / 'model'))}: 받은 모델({escape(str(config['training']['output_dir']))}) 대신 임시 폴더에 저장"
)
console.print(
    f"  --limit {RUN_LIMIT}: 질문 {RUN_LIMIT}개만 · --max-steps {RUN_STEPS} · --no-dev-eval: dev 평가(코퍼스 임베딩 2회) 생략\n"
)
seconds = ragkit(
    "train",
    str(CONFIG),
    "--splits",
    str(SPLITS),
    "--output-dir",
    str(tmp / "model"),
    "--max-steps",
    str(RUN_STEPS),
    "--limit",
    str(RUN_LIMIT),
    "--no-dev-eval",
)
# 파일럿의 학습 기록 (seconds 등)
pilot = json.loads((tmp / "model" / "train_meta.json").read_text(encoding="utf-8"))
console.print(
    f"\n학습 [bold]{pilot['seconds']:.0f}초[/] (step당 [bold]{pilot['seconds'] / RUN_STEPS:.1f}초[/], 모델 로드 포함 전체 {seconds:.0f}초)",
    highlight=False,
)
console.print(
    f"  같은 속도로 train 전체 3 epoch를 돌리면 약 {pilot['seconds'] / RUN_STEPS * with_neg / 32 * 3 / 60:.0f}분"
    f" (받은 모델의 실제 학습은 {meta['seconds'] / 60:.0f}분, epoch마다 dev 평가와 저장이 더해진다)"
)

console.print("\n같은 배치를 파일럿 모델로 다시 재면:")
pilot_embed = create_embedding_fn(
    str(tmp / "model"), checkpoint_path=tmp / "model", backend="torch"
)  # 임시 폴더의 파일럿 모델을 torch로 바로 읽는다 (ONNX 변환 없이)
after_scores, after_loss = in_batch_scores(pilot_embed, batch, by_id)
show_matrix(after_scores, after_loss)

# 질문마다 정답 점수 - 가장 높은 오답 점수
before_margin, after_margin = batch_margins(before_scores), batch_margins(after_scores)
console.print(
    "\n질문별 여유 (정답 점수 - 가장 높은 오답 점수, 0보다 크면 정답이 1위)"
)
for i, (a, b) in enumerate(zip(before_margin, after_margin), 1):
    color = "green" if b > a else "yellow"
    console.print(f"  질문{i}  {a:+.3f} → [{color}]{b:+.3f}[/]", highlight=False)
improved = int((after_margin > before_margin).sum())
console.print(
    f"  [bold]{improved}/{len(batch)}[/]개 질문에서 여유가 커졌다. 배치 손실 {before_loss:.3f} → [bold]{after_loss:.3f}[/]",
    highlight=False,
)
console.print(
    f"  점수 범위: 학습 전 {before_scores.min():.2f}~{before_scores.max():.2f}에 몰려 있던 점수가 "
    f"{after_scores.min():.2f}~{after_scores.max():.2f}로 퍼졌다 (MNRL이 정답과 오답을 떼어 놓는다)"
)
console.print(
    "  확인 세 가지: 손실이 줄었나 · 메모리가 버텼나 · step당 시간으로 전체 시간을 어림했나"
)
shutil.rmtree(tmp, ignore_errors=True)
console.print(
    f"\n[dim]임시 폴더 {escape(str(tmp))}는 지웠다 (모델 약 470MB). 받은 모델은 그대로다.[/]"
)
console.print("다음(6교시): 기본 실험 001을 판정하고 오답을 쪼개 다음 가설을 찾는다")
