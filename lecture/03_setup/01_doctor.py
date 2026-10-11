"""
실습 0-1 (3교시): doctor — 환경과 리소스 점검
=============================================

학습 목표:
- 3~8교시 실습에 필요한 설치 · 모델 · 데이터 · 인덱스를 한 번에 점검한다
- 받은 모델과 인덱스가 실제로 열리고 검색되는지 확인한다 (동작 확인)
- --fix 한 번으로 빠진 것을 모두 받아, 교시마다 따로 받을 일이 없게 한다

동작 확인에서 모델을 한 번씩 열어 검색해 보므로 수십 초 걸린다.

무엇을 하나:
    1. 설치         Python 3.10+, ragkit과 extra(torch · train · mlflow), 앱 4개, 디스크 5GB, Node.js + pnpm
    2. 모델 · 인덱스  학습 전 e5와 ONNX 변환본과 인덱스, 파인튜닝 모델 2개(각 인덱스 포함), 7교시 INT8 · 가지치기 모델
    3. 데이터       코퍼스, 질문 폴더, 법령 단위 분할(data/splits), 조 단위 분할(data/splits_article), 데이터 버전 v1 일치
    4. 동작 확인     학습 전 e5와 002_hard_negatives를 ragkit.service.Searcher.open으로 열어 "야간 근로 수당"을 검색 (--quick이면 생략)
    출력           교시별 준비(✓/✗) → 빠진 것 표 → 받는 법(--fix가 대신 돌릴 명령). 빠진 것이 있으면 종료 코드 1
    --fix          빠진 것만 모아 설치 → 데이터 → 학습 전 모델 → ONNX → Drive 순서로 받고, 끝나면 다시 점검한다

사전 준비:
    uv sync --all-packages --all-extras        # ragkit + 모든 앱 + 학습 도구

실행:
    uv run python lecture/03_setup/01_doctor.py --fix     # 빠진 것을 받고 다시 점검 (처음 한 번, 약 1.5GB)
    uv run python lecture/03_setup/01_doctor.py           # 점검만 (파일 + 동작 확인)
    uv run python lecture/03_setup/01_doctor.py --quick   # 파일과 패키지만 (몇 초)
    uv run python lecture/03_setup/01_doctor.py --all     # 다 된 항목까지 전부 보기
"""

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import cyclopts
from cyclopts import Parameter
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ragkit.config import get_settings
from ragkit.retrieval import default_index_path, model_key
from ragkit.training import data_version as dv

ROOT = Path(__file__).resolve().parents[2]
console = Console(emoji=False)
settings = get_settings()
DATA = settings.data_dir
MODELS = settings.models_dir
E5 = settings.embedding_model_name  # intfloat/multilingual-e5-small
E5_DIR = MODELS / Path(E5).name
FINETUNED = MODELS / "finetuned"
DATA_VERSION = "v1"
ALL = {3, 4, 5, 6, 7, 8}  # 교시. 1 · 2교시는 이론이라 점검할 것이 없다

# 교시 → (실습 번호, 내용). 4교시는 실습 번호 없이 선택 실습 하나
PERIODS = {
    3: ("실습 0", "환경 · 완성품 · 학습 전후 비교"),
    4: ("선택", "질문 데이터 보기"),
    5: ("실습 1", "데이터 · 평가 · 기본 실험"),
    6: ("실습 2", "001 판정 · 오답 분석 · 002 기록"),
    7: ("실습 3", "ONNX · INT8 · 가지치기 · 서빙"),
    8: ("실습 4", "CLI · 스킬 · MCP · 웹"),
}

# 받는 명령. --fix는 이 순서로 돌린다 (설치 → 데이터 → 학습 전 모델 → ONNX → Drive)
SYNC = "uv sync --all-packages --all-extras"
GET_DATA = f"uv run python scripts/data_version.py pull {DATA_VERSION}"
# 5교시 조 단위 분할: 질문에서 바로 만든다 (seed 42, 실험 001과 같은 분할)
SPLIT_ARTICLE = "uv run ragkit split data/questions --unit article --test-ratio 0.15 --dev-ratio 0.1 --out data/splits_article"
GET_E5 = "uv run python scripts/download_model_hf.py"
EXPORT_ONNX = f"uv run ragkit export-onnx models/{E5_DIR.name}"
FROM_DRIVE = "uv run python scripts/finetuned_drive.py download"
FIX_ORDER = (SYNC, GET_DATA, SPLIT_ARTICLE, GET_E5, EXPORT_ONNX, FROM_DRIVE)

# 7교시: 가지치기 · 양자화한 모델과 그 인덱스. Drive의 묶음 하나(optimize)로 받는다
OPTIMIZED = [
    MODELS / f"{E5_DIR.name}-int8" / "onnx" / "model.onnx",
    MODELS / f"{E5_DIR.name}-pruned" / "onnx" / "model.onnx",
    MODELS / f"{E5_DIR.name}-pruned-int8" / "onnx" / "model.onnx",
    default_index_path(f"{E5_DIR.name}-int8"),
    default_index_path(f"{E5_DIR.name}-pruned-int8"),
]


@dataclass
class Check:
    """점검 항목 하나."""

    name: str
    ok: bool
    detail: str = ""
    fix: str = ""  # uv로 시작하면 --fix가 돌리는 명령, 아니면 손으로 할 일
    periods: set[int] = field(default_factory=set)  # 이 항목이 필요한 교시
    optional: bool = False  # 없어도 실습은 되고 일부 단계만 건너뛴다
    drive: str = ""  # Drive에서 받을 이름 (finetuned_drive.py download <이름>)

    @property
    def auto(self) -> bool:
        return self.fix.startswith("uv ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def has_modules(*names: str) -> bool:
    return all(importlib.util.find_spec(n) is not None for n in names)


def version_of(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return ""


def size_mb(path: Path) -> str:
    if path.is_file():
        return f"{path.stat().st_size / 1e6:,.0f} MB"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return f"{total / 1e6:,.0f} MB"


def jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ============================================================
# 점검 항목
# ============================================================
# 항목 하나가 Check 하나다. 설치 · 모델 · 데이터는 패키지와 파일이 있는지만 본다(몇 초).
# periods는 그 항목이 필요한 교시라, 출력의 "교시별 준비"에서 교시마다 빠진 것을 센다.
# fix가 "uv "로 시작하면 --fix가 대신 돌리고, 아니면 "손으로 할 것"에 적는다.
def install_checks() -> list[Check]:
    checks = [
        Check("Python 3.10+", sys.version_info >= (3, 10), sys.version.split()[0], "uv python install 3.12", ALL),
        Check("ragkit (ONNX 추론)", has_modules("ragkit", "onnxruntime", "sqlite_vec"), version_of("ragkit"), SYNC, ALL),  # 검색의 최소 조건
        Check("받기 도구 (gdown, huggingface-hub)", has_modules("gdown", "huggingface_hub"), version_of("gdown"), SYNC, {3}),  # Drive · HF 받기
        Check("extra [torch]", has_modules("torch", "transformers"), version_of("torch"), SYNC, {3, 5, 6, 7, 8}),  # 파인튜닝 모델은 ONNX가 없어 torch로 연다
        Check("extra [train]", has_modules("sentence_transformers", "peft", "datasets", "onnx"), version_of("sentence-transformers"), SYNC, {3, 5, 6, 7}),  # 5교시 파일럿 학습, 7교시 ONNX 변환
        Check("extra [mlflow]", has_modules("mlflow"), version_of("mlflow"), SYNC, {6, 8}, optional=True),  # 선택 실습(모니터링)에만 쓴다
    ]  # fmt: skip
    for dist, module, periods in (
        ("ragkit-search", "ragkit_search", {3, 8}),
        ("ragkit-api", "ragkit_api", {7, 8}),
        ("ragkit-bench", "ragkit_bench", {7}),
        ("ragkit-mcp", "ragkit_mcp", {8}),
    ):
        checks.append(
            Check(f"앱 {dist}", has_modules(module), version_of(dist), SYNC, periods)
        )
    # --fix가 받는 것이 약 1.5GB, 7교시 변환 산출물까지 여유를 둔다
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    checks.append(
        Check(
            "디스크 여유 5GB 이상",
            free_gb >= 5,
            f"{free_gb:,.0f} GB 남음",
            "다른 파일을 정리해 5GB를 비운다",
            ALL,
        )
    )
    node, pnpm = shutil.which("node"), shutil.which("pnpm")
    checks.append(
        Check(
            "Node.js + pnpm (웹 화면)",
            bool(node and pnpm),
            "있음"
            if node and pnpm
            else f"node {'있음' if node else '없음'}, pnpm 없음",
            "Node.js 20+ 설치 후 corepack enable (또는 brew install pnpm)",
            {8},
        )
    )
    return checks


def finetuned_check(name: str, label: str, periods: set[int]) -> Check:
    """파인튜닝 모델과 그 인덱스. Drive에서 둘을 함께 받는다."""
    path = FINETUNED / name
    have_model = (path / "model.safetensors").exists()
    # 인덱스 파일 이름은 폴더 이름 + 가중치 수정 시각(model_key). 모델만 있고 인덱스가 없으면 "인덱스 없음"
    have_index = have_model and default_index_path(model_key(name, path)).exists()
    detail = (
        ""
        if have_model and have_index
        else ("인덱스 없음" if have_model else "모델 없음")
    )
    return Check(
        f"{rel(path)} + 인덱스 ({label})",
        have_model and have_index,
        detail,
        FROM_DRIVE,
        periods,
        drive=name,
    )


def model_checks() -> list[Check]:
    # 학습 전 e5 인덱스: data/processed/index/multilingual-e5-small.sqlite
    base_index = default_index_path(model_key(E5))
    optimized_missing = [p for p in OPTIMIZED if not p.exists()]
    # 학습 전 e5는 HF에서 받고 ONNX는 직접 변환한다. 인덱스는 만드는 데 몇 분이라 Drive에서 받는다
    return [
        Check(f"{rel(E5_DIR)} (학습 전 e5)", (E5_DIR / "config.json").exists(), "", GET_E5, ALL - {4}),
        Check(f"{rel(E5_DIR)}/onnx (ONNX 추론용)", (E5_DIR / "onnx" / "model.onnx").exists(), "", EXPORT_ONNX, ALL - {4}),
        Check(f"{rel(base_index)} (학습 전 e5 인덱스)", base_index.exists(), "", FROM_DRIVE, ALL - {4}, drive=E5_DIR.name),
        finetuned_check("002_hard_negatives", "최종 모델", {3, 6, 8}),
        finetuned_check("001_baseline", "기본 실험", {5, 6}),
        Check(
            "INT8 · 가지치기 모델 3개 + 인덱스 2개",
            not optimized_missing,
            f"{len(optimized_missing)}개 없음 (예: {rel(optimized_missing[0])})" if optimized_missing else "",
            FROM_DRIVE,
            {7},
            drive="optimize",
        ),
    ]  # fmt: skip


def data_checks() -> list[Check]:
    # 검색 대상 조문 조각 전체. 모든 교시가 쓴다
    corpus = DATA / "processed" / "law_docs.json"
    # 법령 · 파트별 질문 파일. 4교시에서 연다
    questions = sorted((DATA / dv.QUESTIONS_DIR).glob("*.jsonl"))
    checks = [
        Check(rel(corpus), corpus.exists(), "", GET_DATA, ALL),
        Check(f"{rel(DATA / dv.QUESTIONS_DIR)}/", bool(questions), f"파일 {len(questions)}개" if questions else "", GET_DATA, {4}),
    ]  # fmt: skip
    # 법령 단위 분할(data/splits): 7 · 8교시 비교 평가의 test (6교시는 조 단위 분할을 쓴다)
    for name, periods in (("train", set()), ("dev", set()), ("test", {7, 8})):
        path = DATA / "splits" / f"{name}.jsonl"
        detail = f"질문 {jsonl_rows(path):,}개" if path.exists() else ""
        checks.append(Check(rel(path), path.exists(), detail, GET_DATA, periods))
    # 조 단위 분할(data/splits_article): 5교시 기본 실험 001과 6교시 판정 · 오답 분석의 분할. 없으면 --fix가 ragkit split으로 만든다
    for name in ("train", "dev", "test", "split_meta"):
        path = (
            DATA
            / "splits_article"
            / f"{name}.{'json' if name == 'split_meta' else 'jsonl'}"
        )
        detail = (
            f"질문 {jsonl_rows(path):,}개"
            if path.exists() and name != "split_meta"
            else ""
        )
        checks.append(Check(rel(path), path.exists(), detail, SPLIT_ARTICLE, {5, 6}))

    # 데이터 버전 v1의 파일 해시와 비교한다. 내가 만든 질문 · 분할이 있으면 달라도 되므로 선택 항목
    manifest_path = dv.versions_dir(DATA) / f"{DATA_VERSION}.json"
    if manifest_path.exists():
        diffs = dv.local_diff(dv.read_manifest(manifest_path), DATA)
        checks.append(
            Check(
                f"데이터 버전 {DATA_VERSION}과 같은가",
                not diffs,
                "일치" if not diffs else f"다름 {len(diffs)}건 (예: {diffs[0]})",
                f"{GET_DATA} --force 로 되돌린다 (내가 만든 질문 · 분할이면 그대로 둬도 된다)",
                {5, 6},
                optional=True,
            )
        )
    return checks


def smoke_checks() -> list[Check]:
    """받은 모델과 인덱스를 실제로 열어 검색해 본다 (파일이 있어도 깨졌거나 짝이 안 맞을 수 있다)."""
    from ragkit.service import Searcher  # 앱(api · search-cli · mcp)과 같은 검색 입구

    corpus = DATA / "processed" / "law_docs.json"
    n_docs = (
        len(json.loads(corpus.read_text(encoding="utf-8"))) if corpus.exists() else None
    )  # 코퍼스 조각 수. 인덱스의 문서 수와 같아야 같은 버전으로 만든 것이다
    # Searcher.open() 기본값: e5 + ONNX + 그 인덱스
    targets = [("학습 전 e5", {}, ALL - {4})]
    if (FINETUNED / "002_hard_negatives" / "model.safetensors").exists():
        targets.append(
            (
                "파인튜닝 002_hard_negatives",
                {"checkpoint": FINETUNED / "002_hard_negatives"},
                {3, 6, 8},
            )
        )
    checks = []
    for label, kwargs, periods in targets:
        name = f"{label}: 열고 검색하기"
        start = time.perf_counter()
        try:
            # 모델 로드 + 인덱스 열기. 모델 하나에 몇 초
            searcher = Searcher.open(**kwargs)
            # 02_try_product.py의 첫 질문과 같다
            hits = searcher.search("야간 근로 수당", k=3)
        except Exception as e:  # noqa: BLE001 — 어떤 이유든 그대로 보여 준다
            detail = f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"
            checks.append(
                Check(name, False, detail, "위 명령으로 모델 · 인덱스를 받은 뒤에도 이 줄이 남으면 상세의 오류를 본다", periods)
            )
            continue
        took = time.perf_counter() - start
        count = searcher.doc_count
        # 결과가 있고 코퍼스와 문서 수가 같아야 통과
        ok = bool(hits) and (n_docs is None or count == n_docs)
        detail = f"문서 {count:,}개 · {took:.1f}초"
        fix = "인덱스를 다시 받는다"
        if n_docs is not None and count != n_docs:
            detail += f" · 코퍼스 {n_docs:,}개와 다름"
            fix = "코퍼스와 인덱스 버전이 다르다. 둘 다 다시 받는다"
        checks.append(Check(name, ok, detail, fix, periods))
    return checks


def collect(quick: bool) -> dict[str, list[Check]]:
    groups = {
        "설치": install_checks(),
        "모델 · 인덱스": model_checks(),
        "데이터": data_checks(),
    }
    if not quick:  # --quick이면 파일 · 패키지 점검(몇 초)만 하고 모델은 열지 않는다
        console.print("[dim]동작 확인 중: 모델과 인덱스를 열어 검색해 봅니다 (수십 초)...[/]")
        groups["동작 확인"] = smoke_checks()
    return groups


# ============================================================
# 받는 명령
# ============================================================
# --fix는 빠진 항목의 fix 명령을 모아 FIX_ORDER(설치 → 데이터 → 학습 전 모델 → ONNX → Drive) 순서로 돌린다.
# Drive에서 받을 것(학습 전 e5 인덱스 · 파인튜닝 모델 · 7교시 모델)은 이름을 모아 finetuned_drive.py 한 줄로 만든다.
# 이미 있는 항목은 명령에 들어가지 않으므로 두 번 받지 않는다. 한 명령이 실패해도 다음 명령으로 간다.
def fix_commands(missing: list[Check]) -> list[str]:
    """--fix가 돌릴 명령. 같은 명령은 하나로, Drive에서 받을 것은 이름을 모아 한 줄로."""
    # 예: ["002_hard_negatives", "multilingual-e5-small"]
    drive = sorted({c.drive for c in missing if c.drive})
    # 같은 명령(예: uv sync)은 집합이라 하나로
    cmds = {c.fix for c in missing if c.auto and not c.drive}
    if drive:
        cmds.add(f"{FROM_DRIVE} {' '.join(drive)}")
    return sorted(
        cmds,
        key=lambda cmd: next(
            (i for i, p in enumerate(FIX_ORDER) if cmd.startswith(p)), len(FIX_ORDER)
        ),
    )


def run_fixes(cmds: list[str]) -> None:
    for cmd in cmds:
        console.print(f"\n[dim]$ {escape(cmd)}[/]", soft_wrap=True)
        result = subprocess.run(cmd, shell=True, cwd=ROOT, check=False)
        if result.returncode != 0:
            console.print(
                f"  [red]실패[/] (종료 코드 {result.returncode}). 이어서 다음 명령을 실행합니다."
            )


# ============================================================
# 출력
# ============================================================
# 기본 출력은 "교시별 준비" 여섯 줄과 빠진 것뿐이다. ✓ 준비됨 · ✗ 빠짐 · △ 선택(없어도 실습은 된다).
# 빠진 것이 있으면 표(이름 · 필요한 교시 · 설명)와 "받는 법"(--fix가 돌릴 명령, 손으로 할 일)을 덧붙인다.
# --all이면 설치 · 모델 · 데이터 · 동작 확인 그룹별로 다 된 항목까지 보여 준다.
def mark(check: Check) -> str:
    """통과 ✓ 초록, 선택 항목 미비 △ 노랑, 필수 미비 ✗ 빨강."""
    if check.ok:
        return "[green]✓[/]"
    return "[yellow]△[/]" if check.optional else "[red]✗[/]"


def missing_rows(missing: list[Check]) -> list[tuple[str, str, str]]:
    """빠진 것 표의 줄 (이름, 교시, 설명). 데이터 버전 하나로 받는 파일은 한 줄로 모은다."""
    data = [c for c in missing if c.fix == GET_DATA]
    rows = [(c.name, periods_text(c), c.detail) for c in missing if c.fix != GET_DATA]
    if data:
        merged = Check("", False, periods=set().union(*(c.periods for c in data)))
        names = ", ".join(Path(c.name.rstrip("/")).name for c in data)
        rows.insert(
            0,
            (
                f"데이터 {DATA_VERSION} (코퍼스 · 질문 · 분할)",
                periods_text(merged),
                names,
            ),
        )
    return rows


def periods_text(check: Check) -> str:
    return (
        "모든 교시"
        if check.periods >= ALL - {4}
        else ", ".join(f"{p}교시" for p in sorted(check.periods))
    )


def show_all(groups: dict[str, list[Check]]) -> None:
    """--all: 다 된 항목까지 전부."""
    for title, checks in groups.items():
        console.print(f"\n[bold]{escape(title)}[/]")
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, show_header=False)
        table.add_column("", width=1)
        table.add_column("항목", overflow="fold")
        table.add_column("상세", overflow="fold")
        table.add_column("교시", style="dim")
        for c in checks:
            table.add_row(mark(c), escape(c.name), escape(c.detail), periods_text(c))
        console.print(table)


def rule(title: str) -> None:
    console.print()
    console.rule(f"[bold]{escape(title)}[/]", align="left", style="dim")


def report(groups: dict[str, list[Check]], show_everything: bool) -> list[Check]:
    checks = [c for group in groups.values() for c in group]
    # 이것이 있으면 종료 코드 1
    missing = [c for c in checks if not c.ok and not c.optional]
    # 선택 항목은 따로 보여 주고 종료 코드에 넣지 않는다
    skipped = [c for c in checks if not c.ok and c.optional]

    console.print()
    console.rule("[bold]실습 0-1: doctor — 환경과 리소스 점검[/]", style="blue")
    if show_everything:
        show_all(groups)

    rule("교시별 준비")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, show_header=False)
    table.add_column("", width=1)
    table.add_column("교시")
    table.add_column("실습")
    table.add_column("내용")
    table.add_column("상태")
    for period, (lab, title) in PERIODS.items():
        gaps = [c for c in missing if period in c.periods]
        status = "준비됨" if not gaps else f"[bold]빠진 것 {len(gaps)}개[/]"
        table.add_row(
            "[red]✗[/]" if gaps else "[green]✓[/]", f"{period}교시", lab, title, status
        )
    console.print(table)

    if missing:
        rule("빠진 것")
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, show_header=False)
        table.add_column("", width=1)
        table.add_column("항목", overflow="fold")
        table.add_column("교시", style="dim")
        table.add_column("상세", overflow="fold")
        for name, periods, detail in missing_rows(missing):
            table.add_row("[red]✗[/]", escape(name), periods, escape(detail))
        console.print(table)

        rule("받는 법")
        cmds = fix_commands(missing)
        manual = sorted({c.fix for c in missing if not c.auto})
        if cmds:
            console.print("  [bold]uv run python lecture/03_setup/01_doctor.py --fix[/]")
            console.print("  → 아래를 대신 돌린다 (로그인 필요 없음)")
            for cmd in cmds:
                console.print(f"      [dim]{escape(cmd)}[/]", soft_wrap=True)
        if manual:
            console.print("  손으로 할 것:")
            for text in manual:
                console.print(f"      {escape(text)}")
    else:
        console.print(
            "\n[green]✓[/] 모두 준비됐습니다. 다음: [dim]uv run python lecture/03_setup/02_try_product.py[/]"
        )

    if skipped:
        rule("선택 (없어도 실습은 된다)")
        for c in skipped:
            console.print(f"  [yellow]△[/] {escape(c.name)}: {escape(c.detail)}")
            console.print(f"      [dim]{escape(c.fix)}[/]", soft_wrap=True)
    if not show_everything:
        console.print("\n[dim]다 된 항목까지 전부 보려면 --all[/]")
    return missing


def main(
    *,
    quick: bool = False,
    fix: bool = False,
    show_all: Annotated[bool, Parameter(name="--all")] = False,
) -> None:
    """강의 환경과 리소스를 점검한다

    Parameters
    ----------
    quick
        파일과 패키지만 본다 (동작 확인 생략)
    fix
        빠진 것을 받고 다시 점검한다
    show_all
        다 된 항목까지 전부 보여 준다
    """
    if fix:
        # 받기 전 상태는 파일만 빠르게 본다. 동작 확인은 받은 뒤 한 번만.
        # 이미 있는 항목은 명령에 들어가지 않으므로 두 번 받지 않는다
        groups = collect(quick=True)
        missing = [
            c for group in groups.values() for c in group if not c.ok and not c.optional
        ]
        if cmds := fix_commands(missing):
            console.print(f"빠진 것 [bold]{len(missing)}개[/]를 받습니다.")
            run_fixes(cmds)
            console.print("\n다시 점검합니다.")
    missing = report(collect(quick), show_all)
    # 빠진 것이 있으면 1 (셸 스크립트에서 조건으로 쓸 수 있다)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    cyclopts.run(main)
