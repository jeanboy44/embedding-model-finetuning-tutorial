# Python 설계 가이드 리팩토링 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 저장소의 Python 코드를 `jeanboy-python-design-guide`(+ `jeanboy-cyclopts-guide`)에 맞춘다. CLI는 전부 cyclopts, 타입 힌트 빠짐없이, 파일 I/O는 `with`, 공통 타입은 한 곳에서.

**Architecture:** 동작은 바꾸지 않는 순수 리팩토링이다. 강의 스크립트의 플래그 이름(`--run`, `--lora`, `--check`, `--file`, `--questions`, `--out`, `--quick`, `--fix`)과 기본값을 그대로 지켜 README·장표·스크립트 docstring의 명령이 바뀌지 않게 한다. 강의 스크립트는 위에서 아래로 읽는 교재라 본문을 `main()`으로 감싸지 않고, 인자 해석 한 곳만 cyclopts로 바꾼다.

**Tech Stack:** Python 3.10+, cyclopts 4.25, pydantic v2, ruff, pytest, uv workspace

**Spec:** 별도 스펙 없음. 기준 문서는 `~/.claude/skills/jeanboy-python-design-guide/SKILL.md`, `~/.claude/skills/jeanboy-cyclopts-guide/SKILL.md`.

## Global Constraints

- 작업은 개별 git worktree(`refactor/python-guide` 브랜치)에서 한다. main에서 직접 고치지 않는다.
- 모든 실행은 `uv run ...`.
- CLI 플래그 이름·기본값·종료 코드는 지금과 같아야 한다 (강의 문서의 명령을 바꾸지 않는다).
- 강의 스크립트(`lecture/**`)와 실험 기록(`runs/001_hard_negatives/`)은 자기 완결: 다른 강의 스크립트·다른 run을 import하지 않는다. 공용 헬퍼를 새로 만들지 않는다 (가이드의 "실험 독립성" 원칙을 교재에도 적용).
- `runs/001_hard_negatives/`는 끝난 실험의 기록이다. 인자 해석과 파일 핸들만 고치고 계산 로직은 한 줄도 바꾸지 않는다.
- `lecture/03_setup/00_download_offline_search.py`는 `uv sync` 전에도 도는 표준 라이브러리 전용 스크립트(PEP 723, 의존성 없음)라 제외한다.
- 출력은 지금처럼 `print`를 쓴다. 강의 스크립트·CLI의 화면 출력은 로그가 아니라 결과물이다 (loguru 도입 대상 아님).
- 커밋 메시지 형식: `refactor(<영역>): <한국어 요약>` + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. `--help`: cyclopts `result_action="return_value"`는 `--help`/`--version` 뒤에 `None`을 돌려주고 스크립트를 계속 돌린다. 모든 강의 스크립트가 `--help`에서 본문을 실행하지 않고 종료 코드 0으로 끝나야 한다 → Task 1의 스모크 테스트가 전 스크립트에 대해 확인.
2. 모르는 인자·위치 인자: 지금 argparse는 `foo` 같은 위치 인자를 에러로 낸다. cyclopts에서 옵션 모델을 키워드 전용(`*,`)으로 받지 않으면 `RUN` 위치 인자가 생겨 `foo`가 bool 파싱 에러로 바뀐다 → 스모크 테스트에서 `extra` 토큰이 0이 아닌 종료 코드를 내는지 확인.
3. 기본 실행(플래그 없음): `sys.argv` 직접 검사를 `args.run`으로 바꿀 때 조건 반전(`"--run" not in sys.argv`)을 놓치기 쉽다 → 각 Task에서 `not args.run`으로 옮겼는지 diff로 확인하고, 데이터가 있는 환경에서 기본 실행을 한 번 돌린다.
4. `--no-run` 같은 자동 부정 플래그: cyclopts가 bool마다 `--no-<name>`을 만든다. 해롭지 않으므로 그대로 둔다 (문서에는 적지 않는다).
5. `01_doctor.py`의 종료 코드: 지금 `sys.exit(1 if ...)`로 빠진 항목이 있으면 1을 낸다. `cyclopts.run(main)`의 기본 result_action에서도 `main()` 안의 `sys.exit`가 그대로 전달되는지 확인.

---

## 현황 조사 결과 (2026-10-10, main `1671e90`)

| 가이드 항목 | 현황 | 조치 |
|---|---|---|
| CLI = cyclopts | `src/`, `apps/`, `scripts/` 대부분은 이미 cyclopts. **argparse 11개 · `sys.argv` 직접 검사 8개**가 남음 (아래 표) | Task 1~5 |
| 타입 힌트 | 테스트 밖 누락 44곳 (`ruff --select ANN001,ANN201,ANN202`): `training/train.py` 15, `tracking.py` 7, embeddings 4, 기타 | Task 6 + 각 Task |
| `with`로 리소스 관리 | `open()`을 `with` 없이 쓰는 곳 8곳, 전부 `runs/001_hard_negatives/` (ruff SIM115) | Task 4 |
| DRY (SSoT) | `EmbedFn = Callable[..., np.ndarray]`가 4곳에 중복 정의 (`rag/answer.py`, `retrieval/index.py`, `service.py`, `apps/bench/.../measure.py`) | Task 6 |
| pathlib / f-string / polars / pandas / logging | `os.path` 0건, pandas 0건, `logging` 0건, `%` 포매팅 0건. `.format()`은 전부 템플릿 상수 채우기라 정상 | 없음 |
| 클래스보다 함수 | 남은 일반 클래스는 상태를 가진 것뿐 (아래 표) | 없음 |

### argparse / sys.argv 사용처

| 파일 | 지금 | 플래그 |
|---|---|---|
| `lecture/03_setup/01_doctor.py` | argparse (`main()` 안) | `--quick`, `--fix` |
| `lecture/04_data/01_corpus.py` | argparse (모듈 최상위) | `--run` |
| `lecture/04_data/02_questions.py` | argparse | `--run`, `--file PATH` |
| `lecture/04_data/03_split.py` | argparse | `--run` |
| `lecture/05_evaluate/01_metrics.py` | argparse | `--run` |
| `lecture/05_evaluate/02_query_expansion.py` | argparse | `--run` |
| `lecture/06_train/01_train_taste.py` | argparse | `--run`, `--lora` |
| `lecture/06_train/02_compare_runs.py` | argparse | `--run` |
| `lecture/06_train/03_error_analysis.py` | argparse | `--run` |
| `lecture/07_optimize/02_onnx_quantize_prune.py` | `RUN = "--run" in sys.argv` | `--run` |
| `lecture/07_optimize/03_bench_and_serve.py` | `"--run" in sys.argv` | `--run` |
| `lecture/08_product/01_search_cli.py` | `"--run" not in sys.argv` | `--run` |
| `lecture/08_product/02_agent_skill.py` | argparse | `--run`, `--questions N`, `--out PATH` |
| `lecture/08_product/04_web_app.py` | `"--run"`, `"--check" in sys.argv` | `--run`, `--check` |
| `lecture/08_product/05_monitoring.py` | `CHECK = "--check" in sys.argv` | `--check` |
| `runs/001_hard_negatives/compare_runs.py` | `sys.argv[1], sys.argv[2]` | `<기준 run> <비교 run>` |
| `runs/001_hard_negatives/dump_topk.py` | `sys.argv[1..3]` | `<모델> <split> <출력 폴더>` |
| `runs/001_hard_negatives/mine.py` | `{...}[sys.argv[1]]()` | `candidates` / `finalize-b` |
| `scripts/deck_to_html.py` | argparse (`main()` 안) | `<deck_dir> <out>` |

### 클래스로 남기는 것 (변경 없음, 이유)

| 클래스 | 이유 |
|---|---|
| `retrieval/index.py: VectorIndex` | sqlite 연결·임베딩 함수를 들고 여러 메서드가 공유 |
| `service.py: Searcher` | 모델·인덱스·LLM 클라이언트를 한 번 열어 요청마다 재사용 |
| `apps/api/.../store.py: NotebookStore` | DB 연결 상태 |
| `tracking.py: Run`, `Span` | MLflow가 꺼져 있을 때 아무 일도 안 하는 null object. 메서드 여러 개가 같은 핸들을 공유 |
| `tracking.py: _ModelFolder`, `train.py: BestEpoch` | 프레임워크(mlflow pyfunc, transformers TrainerCallback)가 상속을 요구 |
| `torch_backend.py: Tokenizer(Protocol)` | 타입 정의 |
| `ExpansionUnavailable` | 예외 |
| 나머지 (`*Event`, `SearchHit`, `LawInfo`, `ModelProfile`, `Variant`, `Check`, `LawArticle`, `DatasetRef`, …) | 이미 dataclass / pydantic 모델 |

---

## 강의 스크립트 공통 패턴 (Task 1~3)

강의 스크립트는 위에서 아래로 읽는 교재다. 본문을 함수로 감싸지 않고, argparse 블록 자리에 아래 블록을 넣는다. 이후 코드가 쓰는 `args.run` 같은 이름은 그대로 둔다.

```python
import cyclopts
from cyclopts import Parameter
from pydantic import BaseModel, Field


@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description="<지금 argparse help 문구 그대로>")


def cli(*, options: Options = Options()) -> Options:
    """<지금 ArgumentParser description 그대로>"""
    return options


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)
```

- `*,`(키워드 전용)가 없으면 cyclopts가 `RUN`이라는 위치 인자를 만든다. 반드시 넣는다.
- `if args is None: sys.exit(0)`이 없으면 `--help` 뒤에 본문이 돈다. 반드시 넣는다.
- 파일 위쪽에 `import sys`가 없으면 추가한다.
- help 문구는 Pydantic 모델이라 `Field(description=...)`에 쓴다 (cyclopts 가이드: 모델은 docstring이 아니라 Field).
- 이 동작은 2026-10-10에 cyclopts 4.25.3으로 확인했다: `--run` → `Options(run=True)`, 인자 없음 → `Options(run=False)`, `extra` → "Unused Tokens" 에러(종료 코드 ≠ 0), `--help` → 도움말 출력 후 `None`.

---

### Task 0: 워크트리와 기준선

**Files:** 없음

- [ ] **Step 1: 워크트리 만들기** (superpowers:using-git-worktrees)

```bash
git worktree add ../embedding-model-finetuning-tutorial-refactor -b refactor/python-guide
cd ../embedding-model-finetuning-tutorial-refactor
uv sync --all-packages --all-extras
```

- [ ] **Step 2: 기준선 기록**

```bash
uv run pytest -q 2>&1 | tail -3            # 통과 수를 적어 둔다
uv run ruff check . --statistics -q         # 지금: SIM115 8, RUF100 4, FURB167 3, I001 2, F401 1
uv run ruff check . --select ANN001,ANN201,ANN202 -q --output-format concise | grep -v '^tests/' | wc -l   # 지금: 44
```

---

### Task 1: 강의 스모크 테스트 + 3~5교시 스크립트

**Files:**
- Create: `tests/test_lecture_cli.py`
- Modify: `lecture/03_setup/01_doctor.py:22,550-569`
- Modify: `lecture/04_data/01_corpus.py:23,79-85`
- Modify: `lecture/04_data/02_questions.py:30,88-95`
- Modify: `lecture/04_data/03_split.py:20,70-74`
- Modify: `lecture/05_evaluate/01_metrics.py:24,121-125`
- Modify: `lecture/05_evaluate/02_query_expansion.py:28,167-173`

**Interfaces:**
- Produces: `tests/test_lecture_cli.py`의 `SCRIPTS` 표. Task 2·3이 자기 스크립트를 여기에 추가한다.

- [ ] **Step 1: 실패하는 스모크 테스트 작성**

`--help`는 본문을 돌리지 않고 0으로 끝나야 하고, 문서에 적힌 플래그가 도움말에 보여야 하고, 모르는 위치 인자는 0이 아닌 코드로 끝나야 한다.

```python
"""강의 스크립트의 CLI 표면: --help가 본문 없이 끝나고, 문서의 플래그가 그대로 있는지."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 스크립트 → 강의 문서에 적힌 플래그
SCRIPTS: dict[str, list[str]] = {
    "lecture/03_setup/01_doctor.py": ["--quick", "--fix"],
    "lecture/04_data/01_corpus.py": ["--run"],
    "lecture/04_data/02_questions.py": ["--run", "--file"],
    "lecture/04_data/03_split.py": ["--run"],
    "lecture/05_evaluate/01_metrics.py": ["--run"],
    "lecture/05_evaluate/02_query_expansion.py": ["--run"],
}


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "COLUMNS": "200"},  # 도움말 표가 줄바꿈되어 플래그가 잘리지 않게
    )


@pytest.mark.parametrize(("script", "flags"), SCRIPTS.items())
def test_help_lists_flags_and_exits_zero(script: str, flags: list[str]) -> None:
    result = _run(script, "--help")
    assert result.returncode == 0, result.stderr
    for flag in flags:
        assert flag in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_unknown_positional_is_error(script: str) -> None:
    assert _run(script, "extra").returncode != 0
```

주의: `test_help_lists_flags_and_exits_zero`는 지금 argparse에서도 통과한다 (기준선 테스트). 실패를 확인할 것은 "`--help` 뒤 본문 실행"이므로 Step 3 도중 `if args is None` 줄을 빼고 돌려 실패하는지 한 번 확인한다.

- [ ] **Step 2: 테스트가 지금 코드에서 통과하는지 확인 (기준선)**

Run: `uv run pytest tests/test_lecture_cli.py -v`
Expected: PASS (argparse도 같은 계약을 지킨다)

- [ ] **Step 3: 모듈 최상위 argparse 5개 교체**

각 파일에서 `import argparse`를 지우고 위 "공통 패턴"을 넣는다. 파일별 `Options` 필드와 `cli` docstring:

`lecture/04_data/01_corpus.py`
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description=f"{RUN_THEME} 테마 코퍼스를 임시 폴더에 직접 만들어 비교")


def cli(*, options: Options = Options()) -> Options:
    """실습 1-1: 코퍼스"""
    return options
```

`lecture/04_data/02_questions.py`
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description="틀린 질문 파일로 검증 스크립트 직접 돌려 보기")
    file: Path | None = Field(None, description="--run에서 검증할 내 질문 파일 (스킬로 만든 것)")


def cli(*, options: Options = Options()) -> Options:
    """실습 1-2: 질문 데이터"""
    return options
```

`lecture/04_data/03_split.py`
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description="ragkit split을 임시 폴더에 실행해 비교")


def cli(*, options: Options = Options()) -> Options:
    """실습 1-3: 법령 단위 분할"""
    return options
```

`lecture/05_evaluate/01_metrics.py` (기존 `sys.stdout.reconfigure(...)` 줄은 블록 위에 그대로 둔다)
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description=f"질문 {RUN_SAMPLE}개로 test·dev를 직접 평가한다")


def cli(*, options: Options = Options()) -> Options:
    """실습 2-1: 평가셋과 지표"""
    return options
```

`lecture/05_evaluate/02_query_expansion.py`
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description=f"키가 있으면 질문 {NEW_EXPANSIONS}개를 새로 확장해 함께 비교")


def cli(*, options: Options = Options()) -> Options:
    """실습 2-2: LLM 쿼리 확장"""
    return options
```

모두 그 아래에:
```python
args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)
```

- [ ] **Step 4: `01_doctor.py`는 `main()`이 이미 있으므로 함수 인자로**

`main()` 하나가 곧 CLI인 경우는 저장소 관례(`scripts/download_model_hf.py`의 `run(main)`)를 따른다.

```python
from cyclopts import run


def main(*, quick: bool = False, fix: bool = False) -> None:
    """강의 환경과 리소스를 점검한다.

    Parameters
    ----------
    quick
        파일과 패키지만 본다 (동작 확인 생략).
    fix
        받을 수 있는 것을 받고 다시 점검한다.
    """
    missing = report(collect(quick))
    if fix and missing:
        run_fixes(missing)
        print("\n다시 점검합니다.")
        missing = report(collect(quick))
    sys.exit(1 if any(not c.optional for c in missing) else 0)


if __name__ == "__main__":
    run(main)
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/test_lecture_cli.py -v`
Expected: 12 passed

- [ ] **Step 6: 기본 실행 확인 (데이터가 받아져 있는 환경)**

```bash
uv run python lecture/03_setup/01_doctor.py --quick; echo "exit=$?"   # 지금과 같은 표, 같은 종료 코드
for f in lecture/04_data/0{1,2,3}_*.py lecture/05_evaluate/01_metrics.py lecture/05_evaluate/02_query_expansion.py; do
  uv run python "$f" > /dev/null || echo "FAIL $f"
done
```
Expected: FAIL 줄 없음. (`--run`은 네트워크·시간이 들어 생략. `args.run` 분기 코드는 이 Task에서 건드리지 않았다.)

- [ ] **Step 7: 커밋**

```bash
git add tests/test_lecture_cli.py lecture/03_setup/01_doctor.py lecture/04_data lecture/05_evaluate
git commit -m "refactor(lecture): 3~5교시 스크립트 인자 해석을 argparse에서 cyclopts로"
```

---

### Task 2: 6교시 스크립트

**Files:**
- Modify: `lecture/06_train/01_train_taste.py:26,118,149-159`
- Modify: `lecture/06_train/02_compare_runs.py:25,156-162`
- Modify: `lecture/06_train/03_error_analysis.py:25,131-137`
- Modify: `tests/test_lecture_cli.py` (`SCRIPTS`에 3개 추가)

**Interfaces:**
- Consumes: Task 1의 `tests/test_lecture_cli.py: SCRIPTS`

- [ ] **Step 1: 테스트 표에 추가**

```python
    "lecture/06_train/01_train_taste.py": ["--run", "--lora"],
    "lecture/06_train/02_compare_runs.py": ["--run"],
    "lecture/06_train/03_error_analysis.py": ["--run"],
```

Run: `uv run pytest tests/test_lecture_cli.py -v -k 06_train` → PASS (기준선)

- [ ] **Step 2: argparse 교체** (공통 패턴, `import argparse` 삭제)

`01_train_taste.py` — 바로 아래 `CONFIG = CONFIGS["004" if args.lora else "002"]` 줄은 그대로 둔다.
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description=f"ragkit train으로 {RUN_STEPS} step만 직접 학습")
    lora: bool = Field(False, description="실험 002 대신 LoRA 실험 004 설정으로 (6교시)")


def cli(*, options: Options = Options()) -> Options:
    """실습 3-1: 학습 원리와 맛보기"""
    return options
```

`02_compare_runs.py`
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description=f"test {RUN_SAMPLE}개 표본으로 비교와 검정을 직접 실행")


def cli(*, options: Options = Options()) -> Options:
    """실습 3-2: 실험 비교"""
    return options
```

`03_error_analysis.py`
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description=f"situation 질문 {RUN_SAMPLE}개를 두 모델로 직접 평가해 분석")


def cli(*, options: Options = Options()) -> Options:
    """실습 3-3: 오답 분석"""
    return options
```

- [ ] **Step 3: `01_train_taste.py:118`의 `embed` 타입 힌트**

```python
from collections.abc import Callable
import numpy as np
...
def <기존 함수 이름>(embed: Callable[[list[str]], np.ndarray], ...) -> ...:
```
(`embed`는 `create_embedding_fn`이 돌려주는 함수다. Task 6에서 `ragkit.embeddings.EmbedFn`이 생기면 그것으로 바꿔도 되지만, 이 Task는 Task 6에 의존하지 않도록 `Callable`로 쓴다.)

- [ ] **Step 4: 확인**

```bash
uv run pytest tests/test_lecture_cli.py -v -k 06_train      # 6 passed
for f in lecture/06_train/0{1,2,3}_*.py; do uv run python "$f" > /dev/null || echo "FAIL $f"; done
uv run python lecture/06_train/01_train_taste.py --lora > /dev/null || echo "FAIL --lora"
uv run ruff check lecture/06_train
```

- [ ] **Step 5: 커밋**

```bash
git add lecture/06_train tests/test_lecture_cli.py
git commit -m "refactor(lecture): 6교시 스크립트 인자 해석을 cyclopts로"
```

---

### Task 3: 7~8교시 스크립트 (`sys.argv` 직접 검사 포함)

**Files:**
- Modify: `lecture/07_optimize/02_onnx_quantize_prune.py:63`
- Modify: `lecture/07_optimize/03_bench_and_serve.py:43,102`
- Modify: `lecture/07_optimize/01_api_review.py:49` (타입 힌트만)
- Modify: `lecture/08_product/01_search_cli.py:31-35,115`
- Modify: `lecture/08_product/02_agent_skill.py:30,54-58`
- Modify: `lecture/08_product/03_mcp_server.py:38,42` (타입 힌트만)
- Modify: `lecture/08_product/04_web_app.py:42,60,118`
- Modify: `lecture/08_product/05_monitoring.py:60`
- Modify: `tests/test_lecture_cli.py`

**Interfaces:**
- Consumes: Task 1의 `SCRIPTS`

- [ ] **Step 1: 테스트 표에 추가**

```python
    "lecture/07_optimize/02_onnx_quantize_prune.py": ["--run"],
    "lecture/07_optimize/03_bench_and_serve.py": ["--run"],
    "lecture/08_product/01_search_cli.py": ["--run"],
    "lecture/08_product/02_agent_skill.py": ["--run", "--questions", "--out"],
    "lecture/08_product/04_web_app.py": ["--run", "--check"],
    "lecture/08_product/05_monitoring.py": ["--check"],
```

Run: `uv run pytest tests/test_lecture_cli.py -v -k "07_ or 08_"`
Expected: `sys.argv` 스크립트 5개는 **FAIL** (`--help`를 모르고 본문을 돈다 / 도움말에 플래그가 없다). `02_agent_skill.py`만 PASS. 이것이 이 Task의 실패하는 테스트다.

- [ ] **Step 2: 상수 블록 바로 아래(첫 `section(...)` 호출 전)에 공통 패턴 추가, `sys.argv` 검사를 `args`로**

`07_optimize/02_onnx_quantize_prune.py` — 63행 `RUN = "--run" in sys.argv`를 아래로 바꾼다. 이후 `RUN` 사용처(96, 125, 149, 175, 225, 266행)는 그대로.
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description="ONNX 변환·양자화·가지치기를 임시 폴더에서 직접 실행")


def cli(*, options: Options = Options()) -> Options:
    """실습 4-2: ONNX 변환 · INT8 양자화 · 어휘 가지치기"""
    return options


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)
RUN = args.run
```
(`description` 문구는 파일 docstring의 "실행:" 절에 있는 `--run` 설명과 맞춘다. 위 문구와 다르면 docstring 쪽을 따른다. `cli` docstring도 파일 첫 줄 제목과 맞춘다.)

`07_optimize/03_bench_and_serve.py` — 43행 `SEARCH_REPEAT = 20` 아래에 공통 패턴(`run: bool`, description은 docstring "실행:" 절의 `--run` 설명), 102행 `if "--run" in sys.argv:` → `if args.run:`

`08_product/01_search_cli.py` — `ENV = {...}` 블록 아래에 공통 패턴, 115행 `if "--run" not in sys.argv:` → `if not args.run:` (**조건 반전 주의**)

`08_product/04_web_app.py` — 42행 `PORT = 8765` 아래에:
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description="apps/web/dist가 있어도 pnpm build를 다시 한다")
    check: bool = Field(False, description="서버를 띄워 확인만 하고 바로 끝낸다 (브라우저 안내 생략)")
```
60행 `if "--run" in sys.argv or not (...)` → `if args.run or not (...)`, 118행 `if "--check" in sys.argv:` → `if args.check:`

`08_product/05_monitoring.py` — 60행 `CHECK = "--check" in sys.argv` 자리에 공통 패턴(`check: bool`, description은 docstring "실행:" 절의 `--check` 설명) + `CHECK = args.check`. 이후 `CHECK` 사용처는 그대로.

`08_product/02_agent_skill.py` — argparse 교체. 187행 이후 `args.run`, `args.out`, `args.questions`는 그대로.
```python
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(False, description="Claude Code로 실제 비교 (비용이 든다)")
    questions: int = Field(2, ge=1, description="--run에서 물을 test 질문 수")
    out: Path = Field(OUT, description="--run 결과 폴더")


def cli(*, options: Options = Options()) -> Options:
    """실습 5-2: 에이전트 스킬"""
    return options
```

각 파일 import 블록에 `import cyclopts`, `from cyclopts import Parameter`, `from pydantic import BaseModel, Field`를 넣고, 더 이상 안 쓰는 `import argparse`는 지운다. `sys`는 `sys.exit` 때문에 남긴다.

- [ ] **Step 3: 타입 힌트 (ruff ANN)**

`07_optimize/01_api_review.py:49` — `lines`가 무엇인지 함수 본문에서 확인해 `list[str]`(또는 실제 타입)로.
`08_product/03_mcp_server.py:38,42` — `result`는 MCP 도구 호출 결과다. 본문이 `.content`/`.structuredContent`를 쓰면 `mcp.types.CallToolResult`, dict로 쓰면 `dict[str, Any]`.

- [ ] **Step 4: 확인**

```bash
uv run pytest tests/test_lecture_cli.py -v                     # 전부 PASS
uv run ruff check lecture --select ANN001,ANN201,ANN202,F401
uv run python lecture/07_optimize/02_onnx_quantize_prune.py > /dev/null || echo FAIL
uv run python lecture/07_optimize/03_bench_and_serve.py > /dev/null || echo FAIL
uv run python lecture/08_product/01_search_cli.py > /dev/null || echo FAIL
uv run python lecture/08_product/02_agent_skill.py > /dev/null || echo FAIL
uv run python lecture/08_product/04_web_app.py --check > /dev/null || echo FAIL   # dist가 있어야 한다
uv run python lecture/08_product/05_monitoring.py --check > /dev/null || echo FAIL   # 수 분, mlflow 필요
```

- [ ] **Step 5: 커밋**

```bash
git add lecture/07_optimize lecture/08_product tests/test_lecture_cli.py
git commit -m "refactor(lecture): 7·8교시 sys.argv 직접 검사를 cyclopts 옵션으로"
```

---

### Task 4: `runs/001_hard_negatives` — cyclopts, `with`, 타입 힌트

**Files:**
- Modify: `runs/001_hard_negatives/compare_runs.py:20,26`
- Modify: `runs/001_hard_negatives/dump_topk.py:25`
- Modify: `runs/001_hard_negatives/mine.py:65-74,121-127,152-153`
- Modify: `runs/001_hard_negatives/error_analysis.py:21`

계산 로직은 바꾸지 않는다 (끝난 실험의 기록). 위치 인자 순서·이름, 서브커맨드 이름(`candidates`, `finalize-b`)은 docstring의 "사용:" 줄과 같게 유지한다.

- [ ] **Step 1: `mine.py` — 서브커맨드 두 개를 cyclopts App으로**

```python
from cyclopts import App

app = App(help="001 hard negative 채굴")


@app.command
def candidates() -> None:
    ...


@app.command(name="finalize-b")
def finalize_b() -> None:
    ...


if __name__ == "__main__":
    app()
```
주의: 모듈 상단의 `RUN = Path(__file__).parent`는 이름만 같을 뿐 플래그가 아니다. 그대로 둔다.

- [ ] **Step 2: `mine.py` — 파일 핸들을 `with`로 (SIM115 7곳)**

```python
rows = [json.loads(line) for line in (EMB / "topk_train.jsonl").read_text(encoding="utf-8").splitlines()]
...
(ART / "judge").mkdir(exist_ok=True)
with (
    open(ART / "mined_candidates.jsonl", "w", encoding="utf-8") as out_c,
    open(ART / "negatives_A.jsonl", "w", encoding="utf-8") as out_a,
    open(ART / "judge" / "b_input.jsonl", "w", encoding="utf-8") as out_j,
):
    <기존 for 루프와 그 안의 write 전부를 한 단계 들여쓴다>
```
괄호로 묶은 다중 `with`는 3.10+ 문법이다 (`requires-python >= 3.10`). 기존 코드에 명시적 `.close()`가 있으면 지운다. `finalize_b`의 `labels` 수집 루프(121행), `out`(125행), `mined_candidates.jsonl` 읽기(127행)도 같은 방식으로.

- [ ] **Step 3: `compare_runs.py`, `dump_topk.py` — 위치 인자를 `main()`으로**

```python
from cyclopts import run


def main(base_name: str, other_name: str) -> None:
    """두 run의 dev 실패 유형과 고친/망친 질문을 비교한다.

    Parameters
    ----------
    base_name
        기준 run (예: r001_control_s42).
    other_name
        비교 run (예: r001_A).
    """
    <20행 이후 모듈 최상위 본문>


if __name__ == "__main__":
    run(main)
```
`judge()`처럼 `base_name`/`labels`를 전역으로 읽는 함수가 있으면 인자로 넘기도록 시그니처를 바꾼다. `compare_runs.py:26`의 `for line in open(...)`도 `with open(...) as f: for line in f:`로.

`dump_topk.py`:
```python
def main(model: str, split: Literal["dev", "train"], out: Path) -> None:
    """한 분할의 질문마다 상위 50개 검색 결과와 임베딩을 저장한다.

    Parameters
    ----------
    model
        모델 이름 또는 폴더.
    split
        dev 또는 train.
    out
        출력 폴더.
    """
```

- [ ] **Step 4: `error_analysis.py:21` 타입 힌트와 `with`**

```python
def rows(split: str) -> list[dict]:
    with open(ART / f"topk_{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
```

- [ ] **Step 5: 확인**

```bash
uv run ruff check runs --select SIM115,ANN001,ANN201,ANN202,F401   # 0건
uv run python runs/001_hard_negatives/mine.py --help                # candidates, finalize-b 보임
uv run python runs/001_hard_negatives/compare_runs.py --help
uv run python runs/001_hard_negatives/dump_topk.py --help
uv run python runs/001_hard_negatives/dump_topk.py m bogus /tmp/x; echo "exit=$?"   # split 검증 에러, ≠0
```
`artifacts/`가 남아 있으면(`runs/*/artifacts/`는 gitignore) `compare_runs.py r001_control_s42 r001_A`를 리팩토링 전후로 돌려 표준 출력을 `diff`한다. 같아야 한다.

- [ ] **Step 6: 커밋**

```bash
git add runs/001_hard_negatives
git commit -m "refactor(runs): 001 스크립트 인자를 cyclopts로, 파일 핸들을 with로"
```

---

### Task 5: `scripts/deck_to_html.py`

**Files:**
- Modify: `scripts/deck_to_html.py:12,189-199`

- [ ] **Step 1: argparse → cyclopts**

```python
from cyclopts import run


def main(deck_dir: Path, out: Path) -> None:
    """Slides 덱 소스를 HTML 파일 하나로 묶는다.

    Parameters
    ----------
    deck_dir
        deck.json과 slides/가 있는 폴더.
    out
        출력 HTML 경로.
    """
    out.write_text(build(deck_dir), encoding="utf-8")
    order = json.loads((deck_dir / "deck.json").read_text(encoding="utf-8"))["order"]
    print(f"{out} ({len(order)}장)")


if __name__ == "__main__":
    run(main)
```
`import argparse` 삭제.

- [ ] **Step 2: 확인 — 지금 커밋된 HTML을 그대로 다시 만드는지**

덱 `project/` 폴더가 로컬에 있을 때(`.deck-tmp/` 등):
```bash
uv run python scripts/deck_to_html.py <project 폴더> /tmp/out.html
diff -q /tmp/out.html lecture/01_introduction/slides/01_introduction.html
```
없으면 `--help`로 위치 인자 두 개가 보이는지만 확인한다.

- [ ] **Step 3: 커밋**

```bash
git add scripts/deck_to_html.py
git commit -m "refactor(scripts): deck_to_html 인자 해석을 cyclopts로"
```

---

### Task 6: `EmbedFn` 한 곳으로 + 라이브러리 타입 힌트

**Files:**
- Modify: `src/ragkit/embeddings/__init__.py` (`EmbedFn` 정의, `create_embedding_fn` 반환 타입, `__getattr__`)
- Modify: `src/ragkit/embeddings/onnx_backend.py:38`, `st_backend.py:16`, `torch_backend.py:81`
- Modify: `src/ragkit/rag/answer.py:18`, `src/ragkit/retrieval/index.py:26`, `src/ragkit/service.py:33`, `apps/bench/src/ragkit_bench/measure.py:14,19`
- Modify: `src/ragkit/evaluation/__init__.py:100`, `src/ragkit/models/vocab_prune.py:29`, `src/ragkit/models/embedding_loader.py:41`
- Modify: `src/ragkit/tracking.py:36,61,144,286,298,302`
- Modify: `src/ragkit/training/train.py:82,109,118,138`
- Modify: `apps/api/src/ragkit_api/app.py:78`, `apps/mcp/src/ragkit_mcp/server.py:56`

**Interfaces:**
- Produces: `ragkit.embeddings.EmbedFn = Callable[..., np.ndarray]` (기존 4곳 정의와 같은 타입)

- [ ] **Step 1: SSoT 정의**

`src/ragkit/embeddings/__init__.py`:
```python
from collections.abc import Callable

import numpy as np

EmbedFn = Callable[..., np.ndarray]
"""텍스트 목록을 받아 (n, dim) L2 정규화 임베딩을 돌려주는 함수. create_embedding_fn이 만든다."""
```
`__all__`이 있으면 `"EmbedFn"`을 추가한다. `create_embedding_fn`과 세 백엔드의 `create_*_embedding_fn`에 `-> EmbedFn`.

순환 import 주의: 백엔드 모듈은 `ragkit.embeddings` 패키지 안에 있으므로 `from ragkit.embeddings import EmbedFn` 대신 `__init__`이 백엔드를 import하는 순서를 확인한다. 순환이 생기면 `src/ragkit/embeddings/types.py`에 정의하고 `__init__`에서 다시 내보낸다.

- [ ] **Step 2: 중복 정의 4곳을 import로 교체**

```python
from ragkit.embeddings import EmbedFn
```
(`rag/answer.py`, `retrieval/index.py`, `service.py`, `ragkit_bench/measure.py`에서 `EmbedFn = Callable[..., np.ndarray]` 줄과, 그 때문에만 있던 `Callable` import를 지운다.) `evaluation/__init__.py:101`의 `Callable[[list[str]], np.ndarray]`도 `EmbedFn`으로.

`retrieval/index.py`가 `ragkit.embeddings`를 import하면 무거운 백엔드를 끌어오는지 확인한다. `__init__`이 `__getattr__`로 지연 로딩하므로(110행) 괜찮을 것이나, `uv run python -X importtime -c "import ragkit.retrieval" 2>&1 | grep -E "torch|onnxruntime"`가 비어 있어야 한다.

- [ ] **Step 3: 나머지 타입 힌트**

| 위치 | 힌트 |
|---|---|
| `measure.py:19`, `evaluation/__init__.py:100` `index` | `VectorIndex` (`from ragkit.retrieval import VectorIndex`, 순환이면 `TYPE_CHECKING` 블록) |
| `vocab_prune.py:29` `tokenizer` | `tokenizers.Tokenizer` |
| `embedding_loader.py:41` `_import_torch` | `-> ModuleType` (`from types import ModuleType`) |
| `tracking.py:36` `_mlflow` | `-> ModuleType` |
| `tracking.py:61,144` `mlflow_module`, `mlflow` | `ModuleType \| None`, `ModuleType` |
| `tracking.py:286` `predict(self, context, ..., params=None)` | `context: Any`, `params: dict[str, Any] \| None = None` (mlflow 시그니처) |
| `tracking.py:298,302` `Span.__init__(live)`, `live` 프로퍼티 | `live: Any = None`, `-> Any` (mlflow LiveSpan; mlflow는 선택 의존성이라 Any) |
| `train.py:82` `_dev_evaluator(dev_questions, corpus, query_prefix, passage_prefix, name)` | `list[dict]`, `list[dict]`, `str`, `str`, `str = "dev"`, `-> InformationRetrievalEvaluator` (`TYPE_CHECKING` import) |
| `train.py:109` `_best_epoch_callback(model, ...)` | `model: SentenceTransformer`, `-> TrainerCallback` |
| `train.py:118` `on_evaluate(self, args, state, control, metrics=None, **kwargs)` | `TrainingArguments`, `TrainerState`, `TrainerControl`, `dict[str, float] \| None`, `**kwargs: Any`, `-> None` |
| `train.py:138` `_make_loss(config, model)` | `model: SentenceTransformer`, `-> torch.nn.Module` |
| `app.py:78` `lifespan` | `-> AsyncIterator[None]` |
| `server.py:56` `_call(fn, *args, **kwargs)` | `fn: Callable[..., T]`, `*args: Any`, `**kwargs: Any`, `-> T \| str` (본문의 반환을 확인) |

`train.py`의 torch·sentence-transformers 타입은 extra `[train]`에만 있으므로 반드시 `if TYPE_CHECKING:` 블록에서 import하고 `from __future__ import annotations`를 쓴다 (파일에 이미 있는지 먼저 확인).

- [ ] **Step 4: 확인**

```bash
uv run ruff check . --select ANN001,ANN201,ANN202 -q --output-format concise | grep -v '^tests/'   # 0건
uv run mypy src/ragkit/embeddings src/ragkit/tracking.py src/ragkit/training/train.py --ignore-missing-imports
uv run pytest -q   # Task 0 기준선과 같은 통과 수
git grep -n "EmbedFn = " -- '*.py'   # 한 줄만
```

- [ ] **Step 5: 커밋**

```bash
git add src apps
git commit -m "refactor(ragkit): EmbedFn을 embeddings 한 곳에서 정의, 빠진 타입 힌트 보강"
```

---

### Task 7: 마무리

- [ ] **Step 1: 남은 ruff 정리** — `uv run ruff check . --fix`로 RUF100(쓸데없는 noqa)·FURB167·I001·F401 자동 수정분만 적용하고 diff를 훑는다.
- [ ] **Step 2: 전체 검증**

```bash
uv run pytest -q
uv run ruff check .
git grep -n -e "import argparse" -e "sys\.argv" -- '*.py'   # 0건
```
- [ ] **Step 3: 문서** — 플래그가 그대로라 README·강의 README는 바뀌지 않아야 한다. `git grep -n "argparse" -- '*.md'`로 남은 언급만 확인.
- [ ] **Step 4: 커밋 후 브랜치 마무리** (superpowers:finishing-a-development-branch)

```bash
git commit -am "chore: ruff 자동 수정 (unused noqa, import 정렬)"
```

---

## 범위 밖 (이번에 하지 않음)

- 강의 문서의 `uv run python <파일>` → `uv run <파일>` 일괄 변경. 둘 다 `uv run`이고 문서·장표 수십 곳을 건드려 리팩토링과 섞으면 리뷰가 어려워진다. 하려면 별도 PR.
- 테스트 코드의 타입 힌트(ANN 위반 약 390건 중 대부분이 `tests/`의 fixture 인자). 별도 PR.
- `apps/search-cli`의 `SearcherOptions`(dataclass)를 Pydantic으로 바꾸기. 지금도 cyclopts 가이드의 공유 파라미터 패턴(`@Parameter(name="*")`)을 따르고 있고 동작 차이가 없다.
- `lecture/03_setup/00_download_offline_search.py` (의존성 없는 스크립트라 의도적으로 표준 라이브러리만 쓴다).

## 실행 방식 제안

Task 1을 먼저 끝낸 뒤(스모크 테스트 `SCRIPTS` 표를 만든다) Task 2·3·4·5·6은 파일이 겹치지 않아 병렬 서브에이전트로 나눌 수 있다. 겹치는 파일은 `tests/test_lecture_cli.py`(Task 2·3이 표에 줄을 추가)뿐이라, 병렬로 돌릴 때는 Task 2·3을 한 에이전트에 묶는다.
