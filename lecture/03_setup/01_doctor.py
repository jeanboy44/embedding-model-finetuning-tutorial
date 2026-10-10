"""
실습 0-1 (3교시): doctor — 환경과 리소스 점검
=============================================

학습 목표:
- 3~8교시 실습에 필요한 설치 · 모델 · 데이터 · 인덱스를 한 번에 점검한다
- 받은 모델과 인덱스가 실제로 열리고 검색되는지 확인한다 (동작 확인)
- --fix 한 번으로 빠진 것을 모두 받아, 교시마다 따로 받을 일이 없게 한다

Gemini는 호출하지 않는다. 동작 확인에서 모델을 한 번씩 열어 검색해 보므로 수십 초 걸린다.

사전 준비:
    uv sync --all-packages --all-extras        # ragkit + 모든 앱 + 학습 도구

실행:
    uv run python lecture/03_setup/01_doctor.py --fix     # 빠진 것을 받고 다시 점검 (처음 한 번, 약 1.5GB)
    uv run python lecture/03_setup/01_doctor.py           # 점검만 (파일 + 동작 확인)
    uv run python lecture/03_setup/01_doctor.py --quick   # 파일과 패키지만 (몇 초)
    uv run python lecture/03_setup/01_doctor.py --all     # 다 된 항목까지 전부 보기
"""

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ragkit.config import get_settings
from ragkit.retrieval import default_index_path, model_key
from ragkit.training import data_version as dv

ROOT = Path(__file__).resolve().parents[2]
settings = get_settings()
DATA = settings.data_dir
MODELS = settings.models_dir
E5 = settings.embedding_model_name  # intfloat/multilingual-e5-small
E5_DIR = MODELS / Path(E5).name
FINETUNED = MODELS / "finetuned"
DATA_VERSION = "v1"
ALL = {3, 4, 5, 6, 7, 8}  # 교시

# 교시 → (실습 번호, 내용). 4교시는 실습 번호 없이 선택 실습 하나
PERIODS = {
    3: ("실습 0", "환경 · 완성품 · 학습 전후 비교"),
    4: ("선택", "질문 데이터 보기"),
    5: ("실습 1", "데이터 · 평가 · 기준 실험"),
    6: ("실습 2", "002 판정 · 오답 분석 · 001 기록"),
    7: ("실습 3", "ONNX · INT8 · 가지치기 · 서빙"),
    8: ("실습 4", "CLI · 스킬 · MCP · 웹"),
}

# 받는 명령. --fix는 이 순서로 돌린다 (설치 → 데이터 → 학습 전 모델 → ONNX → Drive)
SYNC = "uv sync --all-packages --all-extras"
GET_DATA = f"uv run python scripts/data_version.py pull {DATA_VERSION}"
GET_E5 = "uv run python scripts/download_model_hf.py"
EXPORT_ONNX = f"uv run ragkit export-onnx models/{E5_DIR.name}"
FROM_DRIVE = "uv run python scripts/finetuned_drive.py download"
FIX_ORDER = (SYNC, GET_DATA, GET_E5, EXPORT_ONNX, FROM_DRIVE)

# 7교시: 가지치기 · 양자화한 모델과 그 인덱스. Drive의 묶음 하나(optimize)로 받는다
OPTIMIZED = [
    MODELS / f"{E5_DIR.name}-int8" / "onnx" / "model.onnx",
    MODELS / f"{E5_DIR.name}-pruned" / "onnx" / "model.onnx",
    MODELS / f"{E5_DIR.name}-pruned-int8" / "onnx" / "model.onnx",
    default_index_path(f"{E5_DIR.name}-int8"),
    default_index_path(f"{E5_DIR.name}-pruned-int8"),
]
PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here"}


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


def width(text: str) -> int:
    """터미널에 보이는 폭. 한글은 두 칸이다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, n: int) -> str:
    return text + " " * max(n - width(text), 0)


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
def install_checks() -> list[Check]:
    checks = [
        Check("Python 3.10+", sys.version_info >= (3, 10), sys.version.split()[0], "uv python install 3.12", ALL),
        Check("ragkit (ONNX 추론)", has_modules("ragkit", "onnxruntime", "sqlite_vec"), version_of("ragkit"), SYNC, ALL),
        Check("받기 도구 (gdown, huggingface-hub)", has_modules("gdown", "huggingface_hub"), version_of("gdown"), SYNC, {3}),
        Check("extra [torch]", has_modules("torch", "transformers"), version_of("torch"), SYNC, {3, 5, 6, 7, 8}),
        Check("extra [train]", has_modules("sentence_transformers", "peft", "datasets", "onnx"), version_of("sentence-transformers"), SYNC, {3, 5, 6, 7}),
        Check("extra [mlflow]", has_modules("mlflow"), version_of("mlflow"), SYNC, {6, 8}, optional=True),
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
    base_index = default_index_path(model_key(E5))
    optimized_missing = [p for p in OPTIMIZED if not p.exists()]
    return [
        Check(f"{rel(E5_DIR)} (학습 전 e5)", (E5_DIR / "config.json").exists(), "", GET_E5, ALL - {4}),
        Check(f"{rel(E5_DIR)}/onnx (ONNX 추론용)", (E5_DIR / "onnx" / "model.onnx").exists(), "", EXPORT_ONNX, ALL - {4}),
        Check(f"{rel(base_index)} (학습 전 e5 인덱스)", base_index.exists(), "", FROM_DRIVE, ALL - {4}, drive=E5_DIR.name),
        finetuned_check("r001_A", "최종 모델", {3, 6, 8}),
        finetuned_check("exp_002", "기준 실험", {5, 6}),
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
    corpus = DATA / "processed" / "law_docs.json"
    questions = sorted((DATA / dv.QUESTIONS_DIR).glob("*.jsonl"))
    checks = [
        Check(rel(corpus), corpus.exists(), "", GET_DATA, ALL),
        Check(f"{rel(DATA / dv.QUESTIONS_DIR)}/", bool(questions), f"파일 {len(questions)}개" if questions else "", GET_DATA, {4}),
    ]  # fmt: skip
    for name, periods in (("train", {5, 6}), ("dev", {5}), ("test", {5, 6, 7, 8})):
        path = DATA / "splits" / f"{name}.jsonl"
        detail = f"질문 {jsonl_rows(path):,}개" if path.exists() else ""
        checks.append(Check(rel(path), path.exists(), detail, GET_DATA, periods))
    meta = DATA / "splits" / "split_meta.json"
    checks.append(Check(rel(meta), meta.exists(), "", GET_DATA, {5}))

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


def key_checks() -> list[Check]:
    key = settings.gemini_api_key.strip()
    ok = key not in PLACEHOLDER_KEYS and not key.startswith("your_")
    detail = "설정됨" if ok else ("예시 값 그대로" if key else "비어 있음")
    return [
        Check(
            "GEMINI_API_KEY (.env)",
            ok,
            detail,
            "cp .env.example .env 후 GEMINI_API_KEY에 AI Studio 키를 넣는다. 없으면 LLM 답변 단계만 건너뛴다",
            {3, 7, 8},
            optional=True,
        )
    ]


def smoke_checks() -> list[Check]:
    """받은 모델과 인덱스를 실제로 열어 검색해 본다 (파일이 있어도 깨졌거나 짝이 안 맞을 수 있다)."""
    from ragkit.service import Searcher

    corpus = DATA / "processed" / "law_docs.json"
    n_docs = (
        len(json.loads(corpus.read_text(encoding="utf-8"))) if corpus.exists() else None
    )
    targets = [("학습 전 e5", {}, ALL - {4})]
    if (FINETUNED / "r001_A" / "model.safetensors").exists():
        targets.append(
            ("파인튜닝 r001_A", {"checkpoint": FINETUNED / "r001_A"}, {3, 6, 8})
        )
    checks = []
    for label, kwargs, periods in targets:
        name = f"{label}: 열고 검색하기"
        start = time.perf_counter()
        try:
            searcher = Searcher.open(**kwargs)
            hits = searcher.search("야간 근로 수당", k=3)
        except Exception as e:  # noqa: BLE001 — 어떤 이유든 그대로 보여 준다
            detail = f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"
            checks.append(
                Check(name, False, detail, "위의 모델 · 인덱스를 먼저 채운다", periods)
            )
            continue
        took = time.perf_counter() - start
        count = searcher.doc_count
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
        "API 키": key_checks(),
    }
    if not quick:
        print("동작 확인 중: 모델과 인덱스를 열어 검색해 봅니다 (수십 초)...")
        groups["동작 확인"] = smoke_checks()
    return groups


# ============================================================
# 받는 명령
# ============================================================
def fix_commands(missing: list[Check]) -> list[str]:
    """--fix가 돌릴 명령. 같은 명령은 하나로, Drive에서 받을 것은 이름을 모아 한 줄로."""
    drive = sorted({c.drive for c in missing if c.drive})
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
        print(f"\n$ {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(
                f"  실패 (종료 코드 {result.returncode}). 이어서 다음 명령을 실행합니다."
            )


# ============================================================
# 출력
# ============================================================
def mark(check: Check) -> str:
    if check.ok:
        return "✓"
    return "△" if check.optional else "✗"


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
        print(f"\n[{title}]")
        name_w = max(width(c.name) for c in checks)
        detail_w = max(width(c.detail) for c in checks)
        for c in checks:
            print(
                f"  {mark(c)} {pad(c.name, name_w)}  {pad(c.detail, detail_w)}  {periods_text(c)}"
            )


def rule(title: str) -> None:
    print(f"\n── {title} " + "─" * max(56 - width(title), 4))


def report(groups: dict[str, list[Check]], show_everything: bool) -> list[Check]:
    checks = [c for group in groups.values() for c in group]
    missing = [c for c in checks if not c.ok and not c.optional]
    skipped = [c for c in checks if not c.ok and c.optional]

    print("\n" + "=" * 60)
    print("실습 0-1: doctor — 환경과 리소스 점검")
    print("=" * 60)
    if show_everything:
        show_all(groups)

    rule("교시별 준비")
    for period, (lab, title) in PERIODS.items():
        gaps = [c for c in missing if period in c.periods]
        status = "준비됨" if not gaps else f"빠진 것 {len(gaps)}개"
        print(
            f"  {'✗' if gaps else '✓'} {period}교시  {pad(lab, 6)}  {pad(title, 32)}  {status}"
        )

    if missing:
        rule("빠진 것")
        rows = missing_rows(missing)
        name_w = max(width(name) for name, _, _ in rows)
        periods_w = max(width(periods) for _, periods, _ in rows)
        for name, periods, detail in rows:
            line = f"  ✗ {pad(name, name_w)}  {pad(periods, periods_w)}  {detail}"
            print(line.rstrip())

        rule("받는 법")
        cmds = fix_commands(missing)
        manual = sorted({c.fix for c in missing if not c.auto})
        if cmds:
            print("  uv run python lecture/03_setup/01_doctor.py --fix")
            print("  → 아래를 대신 돌린다 (로그인 필요 없음)")
            for cmd in cmds:
                print(f"      {cmd}")
        if manual:
            print("  손으로 할 것:")
            for text in manual:
                print(f"      {text}")
    else:
        print(
            "\n✓ 모두 준비됐습니다. 다음: uv run python lecture/03_setup/02_try_product.py"
        )

    if skipped:
        rule("선택 (없어도 실습은 된다)")
        for c in skipped:
            print(f"  △ {c.name}: {c.detail}")
            print(f"      {c.fix}")
    if not show_everything:
        print("\n다 된 항목까지 전부 보려면 --all")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="강의 환경과 리소스를 점검한다")
    parser.add_argument(
        "--quick", action="store_true", help="파일과 패키지만 본다 (동작 확인 생략)"
    )
    parser.add_argument(
        "--fix", action="store_true", help="빠진 것을 받고 다시 점검한다"
    )
    parser.add_argument(
        "--all", action="store_true", help="다 된 항목까지 전부 보여 준다"
    )
    args = parser.parse_args()

    if args.fix:
        # 받기 전 상태는 파일만 빠르게 본다. 동작 확인은 받은 뒤 한 번만
        groups = collect(quick=True)
        missing = [
            c for group in groups.values() for c in group if not c.ok and not c.optional
        ]
        if cmds := fix_commands(missing):
            print(f"빠진 것 {len(missing)}개를 받습니다.")
            run_fixes(cmds)
            print("\n다시 점검합니다.")
    missing = report(collect(args.quick), args.all)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
