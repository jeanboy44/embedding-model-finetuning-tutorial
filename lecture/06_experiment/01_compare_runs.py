"""
실습 2-1 (6교시): 가설 표와 001 판정, 정말 올랐나
==================================================

학습 목표:
- 기본 실험 001 위에 세운 가설(002 오답 다시 캐기 · 003 오답 3개 · 004 · 005 맥락 헤더 · 006 model soup · 007 구어 질문)과 결과를 한 표로 본다
- 학습 전 e5-small과 001을 같은 test(조 단위 분할)로 재고, 차이가 우연인지 질문 단위 대응 검정(ragkit paired-test)으로 판정한다
- 평균이 올라도 새로 틀린 질문이 있다는 것을 숫자로 본다

학습은 하지 않는다. 001은 5교시에 받은 모델(models/finetuned/001_baseline)을 쓴다.

무엇을 하나:
    1. 가설 표를 출력한다 (실험 001~007의 가설 · 결과 · 판정, 006 model soup을 뺀 이유, 판정 규칙)
    2. 학습 전 e5-small과 001의 조 단위 test(1,863개) 결과를 읽는다.
       받은 결과(experiments/results/001_baseline/)가 있으면 그것을, 없으면 ragkit evaluate --splits data/splits_article로
       직접 평가해 experiments/results/lecture/06_experiment/001_baseline/에 쓴다 (모델 하나에 약 1분)
    3. 두 결과를 질문 단위로 짝지어 검정한다 (ragkit paired-test --judge doc): Δ · 95% CI · 고침/망침 · 나빠진 법령 수

사전 준비 (없으면 스크립트가 받는 명령을 알려 주고 끝난다):
    uv run python scripts/data_version.py pull v1                     # 코퍼스 + 질문
    uv run python lecture/05_data_eval/02_split.py                    # 조 단위 분할 data/splits_article/ (없으면 만든다)
    uv run python scripts/finetuned_drive.py download 001_baseline    # 001 모델 + 인덱스 (5교시에 받음, 직접 평가할 때만 필요)

실행:
    uv run python lecture/06_experiment/01_compare_runs.py          # test 1,863개로 판정 (받은 결과가 있으면 몇 초, 없으면 약 2분)
    uv run python lecture/06_experiment/01_compare_runs.py --run    # test 300개 표본으로 평가와 검정을 직접 (1분 안팎)

기본 모드 결과는 experiments/results/lecture/06_experiment/001_baseline/ 에 쓴다(02_error_analysis.py가 이어서 읽는다).
강사가 실험 001 결과 JSON(experiments/results/001_baseline/{000_base_test,001_test}.json)을 나눠 줬으면 그것을 읽는다.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cyclopts
from cyclopts import Parameter
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ragkit.data import load_questions

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
SPLITS = (
    DATA / "splits_article"
)  # 조 단위 분할 (5교시 02_split.py). 법령 단위 data/splits는 쓰지 않는다
TEST = SPLITS / "test.jsonl"
RECEIVED = (
    ROOT / "experiments" / "results" / "001_baseline"
)  # 강사가 나눠 준 실험 001 결과 JSON
OUT = ROOT / "experiments" / "results" / "lecture" / "06_experiment" / "001_baseline"
MODELS = ROOT / "models"
BASE_NAME, TUNED_NAME = "multilingual-e5-small", "001_baseline"
BASE_MODEL = "intfloat/multilingual-e5-small"
TUNED_DIR = MODELS / "finetuned" / TUNED_NAME
RUN_SAMPLE = 300  # --run에서 쓸 test 질문 수
SEED = 42

GET_DATA = "uv run python scripts/data_version.py pull v1"
GET_SPLITS = "uv run python lecture/05_data_eval/02_split.py"
GET_001 = "uv run python scripts/finetuned_drive.py download 001_baseline"

# 우리가 세운 가설과 결과. 숫자는 실험 기록에서 옮겼다
# (test: 실험 001 README, dev: experiments/README.md 표 · 각 실험의 04_results.md).
# 001만 test로 판정했고 002부터는 dev로만 골랐다. 006 model soup은 설계만 하고 돌리지 않았다
HYPOTHESES = [
    (
        "001 기본 실험",
        "우리 데이터로 학습하면 오른다",
        "test 0.744 (학습 전 0.545)",
        "오늘 직접 판정",
    ),
    (
        "002 오답 다시 캐기",
        "001이 헷갈리는 조문 1개를 오답으로 주면 오른다",
        "dev 대조군 0.7144 → 0.7534 (Δ +0.0390)",
        "맞음 · 채택 (03에서 기록 보기)",
    ),
    (
        "003 오답 3개 · 2차 채굴",
        "오답을 3개로 늘리고 002로 다시 캐면 더 오른다",
        "dev B3 0.7526 · A3 0.7586 (대조군 0.7534)",
        "틀림 · 기각",
    ),
    (
        "004 맥락 헤더 제로샷",
        "조문 앞에 [법령 > 장]을 붙이면 학습 없이 오른다",
        "dev 학습 전 0.5374 · 002 0.7409 (둘 다 하락)",
        "틀림 · 기각",
    ),
    (
        "005 맥락 헤더 파인튜닝",
        "헤더 붙인 조문으로 학습하면 오른다",
        "dev 0.7526 (대조군 0.7534, Δ −0.0008)",
        "틀림 · 기각",
    ),
    (
        "006 model soup",
        "여러 번 학습한 가중치를 평균하면 오른다",
        "-",
        "설계만 하고 뺐다",
    ),
    ("007 구어 질문", "일상 말투 질문을 더 만들어 학습하면 오른다", "-", "진행 중"),
]
SOUP_REASONS = [
    "재료 하나가 학습 1회라, 재료 수만큼 학습 시간이 든다",
    "결과는 같은 e5-small 하나라 작아지지도 빨라지지도 않는다 (배포 쪽 이득 없음)",
    "작은 모델 하나를 한 번 학습해 배포한다는 이 과제의 방향과 맞지 않는다",
]


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
    lines = [f"  - {escape(rel(path))}\n      받기: [dim]{escape(how)}[/]" for path, how in missing]
    lines.append(
        "준비 상태는 [dim]uv run python lecture/03_setup/01_doctor.py[/] 로 한 번에 볼 수 있습니다."
    )
    console.print(Panel("\n".join(lines), title="필요한 산출물이 없습니다", border_style="red"))
    raise SystemExit(1)


def ragkit(*args: str) -> float:
    """ragkit CLI를 저장소 루트에서 실행하고 걸린 시간(초)을 돌려준다."""
    shown = [rel(Path(a)) if a.startswith(str(ROOT)) else a for a in args]
    console.print(f"[dim]$ uv run ragkit {escape(' '.join(shown))}[/]", soft_wrap=True)
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    start = time.perf_counter()
    done = subprocess.run(
        [str(Path(sys.executable).parent / "ragkit"), *args],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if done.returncode != 0:
        console.print(
            f"[red]ragkit {escape(args[0])}이 실패했습니다 (종료 코드 {done.returncode}). 위 메시지를 확인하세요.[/]"
        )
        raise SystemExit(done.returncode)
    return time.perf_counter() - start


def load_results(folder: Path) -> dict[str, Path]:
    """결과 폴더의 모델별 JSON을 모델 폴더 이름 → 경로로 찾는다.

    ragkit evaluate 결과(per_question이 있는 JSON)만 고른다. paired_test.json · leak_check.json은 건너뛴다.
    모델 이름은 result["model"]의 마지막 경로 조각이라 받은 파일 이름(000_base_test.json 등)과 무관하다.
    """
    found = {}
    for path in sorted(folder.glob("*.json")):
        if path.name == "comparison.json" or path.stem.endswith("_expand"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if "per_question" in result:
            found[Path(result["model"]).name] = path
    return found


def evaluate_both(splits: Path, out_dir: Path) -> dict[str, Path]:
    """학습 전과 001을 같은 test로 평가해 결과 JSON 경로를 돌려준다.

    ragkit evaluate는 모델별 코퍼스 인덱스(data/processed/index/*.sqlite)가 있으면 재사용하고
    질문만 임베딩한다. 받은 인덱스가 있으면 모델 하나에 1분 안팎이다.
    """
    require([(TUNED_DIR, GET_001)])  # 직접 평가할 때만 001 모델이 필요하다
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, model in ((BASE_NAME, BASE_MODEL), (TUNED_NAME, str(TUNED_DIR))):
        out = out_dir / f"{name}.json"
        seconds = ragkit(
            "evaluate",
            model,
            "--split",
            "test",
            "--splits",
            str(splits),
            "--out",
            str(out),
        )
        console.print(f"  [dim]({seconds:.0f}초)[/]\n")
        paths[name] = out
    return paths


sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
console = Console(emoji=False)  # 데이터 문장의 :...: 이 이모지로 바뀌지 않게


@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(
        False,
        description=f"test {RUN_SAMPLE}개 표본으로 평가와 검정을 직접 실행",
    )


def cli(*, options: Options | None = None) -> Options:
    """실습 2-1: 가설 표와 001 판정"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)

require([(CORPUS, GET_DATA), (TEST, GET_SPLITS)])


# ============================================================
# 1. 우리가 세운 가설
# ============================================================
# 계산 없이 HYPOTHESES 표를 그대로 출력한다. 001은 오늘 직접 판정하고, 002~005는 에이전트가
# dev로만 판정한 기록이다. 판정 규칙(0.02 미만은 차이로 보지 않는다)은 experiments/README.md와 같다
section("1. 우리가 세운 가설: 001 위에 하나씩 바꿔 쌓았다")
table = Table("실험", "가설", "결과", "판정", box=box.SIMPLE_HEAD, show_edge=False)
for name, hypothesis, result, verdict in HYPOTHESES:
    table.add_row(escape(name), escape(hypothesis), escape(result), escape(verdict))
console.print(table)
console.print("  006 model soup을 뺀 이유 (돌리기 전에 비용과 방향을 따졌다)")
for reason in SOUP_REASONS:
    console.print(f"    · {escape(reason)}")
console.print(
    "\n  판정 규칙: 0.02 미만 차이는 차이로 보지 않는다 (dev 표준오차 약 ±0.012, 실험 002 대조군 seed SD 약 0.006)"
)
console.print("  002부터는 dev로만 고르고 test는 쓰지 않았다. 001만 test로 한 번 판정했다")
console.print(
    "  오늘은 001을 직접 판정하고 쪼개 본다. 002의 가설이 어디서 나왔는지는 02에서 찾는다"
)


# ============================================================
# 2. 학습 전 vs 001을 같은 test로
# ============================================================
# 조 단위 test(1,863개)는 54개 법령이 모두 학습에 있고, 학습 질문이 없던 조만 모았다.
# 기본 모드: 받은 결과(RECEIVED) → 이전 실행 결과(OUT) 순으로 찾고, 없으면 ragkit evaluate를 두 번 돌린다.
# --run: test에서 300개를 뽑아 임시 폴더의 splits/test.jsonl로 쓰고 같은 명령으로 평가한다
if args.run:
    section(f"2. --run: test {RUN_SAMPLE}개 표본으로 학습 전 vs 001")
    tmp = Path(tempfile.mkdtemp(prefix="lecture-06-1-"))
    sample = random.Random(SEED).sample(
        load_questions(TEST), RUN_SAMPLE
    )  # seed 고정이라 누가 돌려도 같은 300개
    subset = tmp / "splits" / "test.jsonl"
    subset.parent.mkdir()
    subset.write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in sample),
        encoding="utf-8",
    )
    paths = evaluate_both(subset.parent, tmp / "results")
    out_dir = tmp / "results"
else:
    section(
        "2. 학습 전 vs 001 (조 단위 test 1,863개, 54개 법령 모두 학습에 있고 조만 처음 본다)"
    )
    paths, out_dir = {}, None
    for folder in (RECEIVED, OUT):
        found = load_results(folder)
        if BASE_NAME in found and TUNED_NAME in found:
            paths, out_dir = found, folder
            console.print(
                f"이미 있는 결과를 읽는다: [dim]{escape(rel(folder))}[/] (새로 재려면 [dim]{escape(rel(OUT))}[/]를 지운다)"
            )
            break
    if not paths:
        console.print(
            "두 모델을 test 1,863개로 평가한다. 받은 인덱스가 있으면 질문만 임베딩한다 (모델 하나에 1분 안팎)."
        )
        paths = evaluate_both(SPLITS, OUT)
        out_dir = OUT

base = json.loads(paths[BASE_NAME].read_text(encoding="utf-8"))
tuned = json.loads(paths[TUNED_NAME].read_text(encoding="utf-8"))
console.print(
    f"\n  질문 {base['n']:,}개 · 법령 {len(base['by_law'])}개 · 판정 doc (정답 문서 하나)"
)
table = Table("모델", "R@1", "R@5", "R@10", "MRR@10", box=box.SIMPLE_HEAD, show_edge=False)
for column in table.columns[1:]:
    column.justify = "right"
for label, r in (("학습 전 e5-small", base), ("001 전체 학습", tuned)):
    d = r["doc"]
    table.add_row(
        label,
        f"{d['recall@1']:.3f}",
        f"[bold]{d['recall@5']:.3f}[/]",
        f"{d['recall@10']:.3f}",
        f"{d['mrr@10']:.3f}",
    )
console.print(table)
console.print(
    f"  평균만 보면 [bold]{base['doc']['recall@5']:.3f} → {tuned['doc']['recall@5']:.3f}[/]. 이게 우연이 아닌지는 질문 단위로 본다"
)


# ============================================================
# 3. 차이가 우연인가: 질문 단위 대응 검정
# ============================================================
# ragkit paired-test가 두 결과의 같은 질문끼리 R@5 맞힘/틀림을 짝지어 Δ · 95% CI(부트스트랩) ·
# McNemar p · 고침/망침 수 · 법령별로 나빠진 수를 낸다. CI가 0을 걸치지 않으면 차이로 본다
section("3. 차이가 우연인가: ragkit paired-test (같은 질문끼리 짝지어 비교)")
ragkit("paired-test", str(paths[BASE_NAME]), str(paths[TUNED_NAME]), "--judge", "doc")
console.print("\n  읽는 법")
console.print("    · 95% CI가 0을 걸치지 않으면 차이가 있다고 말한다")
console.print(
    "    · 고침 = 학습 전엔 틀리고 001이 맞힌 질문, 망침 = 그 반대. 평균이 올라도 망친 질문이 있다"
)
console.print(
    "    · 나빠진 법령 수: 특정 법령에서만 오른 것인지 본다 (54개 법령 모두 학습에 있다)"
)
if args.run:
    console.print(
        f"  [yellow]표본 {RUN_SAMPLE}개라 신뢰구간이 전체 test보다 넓다. 같은 차이도 표본이 작으면 확신할 수 없다[/]"
    )

console.print(
    f"\n결과 폴더: [dim]{escape(rel(out_dir))}[/] (모델별 JSON의 per_question = 질문별 순위·top-10)"
)
console.print(
    "다음: 02_error_analysis.py — 어디서 올랐고, 무엇이 여전히 틀리고, 우리가 준 오답은 어땠나"
)
