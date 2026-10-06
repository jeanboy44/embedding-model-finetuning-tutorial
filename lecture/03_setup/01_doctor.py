"""
실습 0-1 (3교시): doctor — 환경과 리소스 점검
=============================================

학습 목표:
- 강의에 필요한 설치(ragkit, extra, 앱, pnpm, 디스크) · 모델 · 데이터 · 인덱스 · API 키를 한 번에 점검한다
- 받은 모델과 인덱스가 실제로 열리고 검색되는지 확인한다 (동작 확인)
- 빠진 것마다 어떤 명령으로 받거나 만드는지 보고, --fix로 받을 수 있는 것은 한 번에 받는다
- 실습 0~5 중 지금 바로 할 수 있는 실습이 어디까지인지 본다

Gemini는 호출하지 않는다. 동작 확인에서 모델을 한 번씩 열어 검색해 보므로 수십 초 걸린다.

사전 준비:
    uv sync --all-packages --all-extras        # ragkit + 모든 앱 + 학습 도구

실행:
    uv run python lecture/03_setup/01_doctor.py           # 전체 점검 (파일 + 동작 확인)
    uv run python lecture/03_setup/01_doctor.py --quick   # 파일과 패키지만 (몇 초)
    uv run python lecture/03_setup/01_doctor.py --fix     # 받을 수 있는 것을 받고 다시 점검
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
from ragkit.rag.query_expansion import default_cache_path
from ragkit.retrieval import default_index_path, model_key
from ragkit.training import data_version as dv

ROOT = Path(__file__).resolve().parents[2]
settings = get_settings()
DATA = settings.data_dir
MODELS = settings.models_dir
E5 = settings.embedding_model_name  # intfloat/multilingual-e5-small
E5_DIR = MODELS / Path(E5).name
FINETUNED = MODELS / "finetuned" / "r001_A"  # 실험 010에서 test R@5가 가장 높은 모델
DATA_VERSION = "v1"

# 받는 명령 (scripts/ 와 ragkit CLI에서 확인한 것)
SYNC = "uv sync --all-packages --all-extras"
GET_E5 = (
    "uv run python scripts/download_model_hf.py"
    "   # 사내망 등 HF가 막히면: uv run python scripts/download_model_gdrive.py"
)
GET_DATA = f"uv run python scripts/data_version.py pull {DATA_VERSION}"
FROM_DRIVE = "강사 Drive에서 받아 {path}에 둔다 (받는 스크립트 없음)"
GET_FINETUNED = (
    "uv run python scripts/finetuned_drive.py download {name}   # 모델 + 인덱스"
)
PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here"}


@dataclass
class Check:
    """점검 항목 하나."""

    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    labs: set[int] = field(default_factory=set)  # 이 항목이 필요한 실습 번호
    optional: bool = False  # 없어도 실습은 되고 일부 단계만 건너뛴다

    def command(self) -> str:
        """--fix로 바로 돌릴 수 있는 명령. 손으로 해야 하거나 실습에서 직접 만드는 것이면 빈 문자열."""
        first = self.fix.splitlines()[0] if self.fix else ""
        cmd = first.split("   ")[0].strip()
        if not cmd.startswith("uv ") or "--force" in cmd or "→" in cmd:
            return ""
        if (
            self.labs == {4} or " ragkit index" in cmd
        ):  # 실습 4에서 직접 만들거나 오래 걸리는 것
            return ""
        return cmd


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
        Check(
            "Python 3.10+",
            sys.version_info >= (3, 10),
            sys.version.split()[0],
            "uv python install 3.12",
            {0, 1, 2, 3, 4, 5},
        ),
        Check(
            "ragkit (기본: ONNX 추론)",
            has_modules("ragkit", "onnxruntime", "sqlite_vec"),
            version_of("ragkit"),
            SYNC,
            {0, 1, 2, 3, 4, 5},
        ),
        Check(
            "데이터 받기 도구 (gdown, huggingface-hub)",
            has_modules("gdown", "huggingface_hub"),
            version_of("gdown"),
            SYNC,
            {0, 1},
        ),
        Check(
            "extra [torch] (torch, transformers)",
            has_modules("torch", "transformers"),
            version_of("torch"),
            SYNC,
            {0, 3, 4},
        ),
        Check(
            "extra [train] (sentence-transformers, peft, onnx)",
            has_modules("sentence_transformers", "peft", "datasets", "onnx"),
            version_of("sentence-transformers"),
            SYNC,
            {3, 4},
        ),
        Check(
            "extra [mlflow]",
            has_modules("mlflow"),
            version_of("mlflow"),
            SYNC,
            {3, 5},
            optional=True,
        ),
    ]
    for dist, module, labs in (
        ("ragkit-api", "ragkit_api", {4, 5}),
        ("ragkit-search", "ragkit_search", {5}),
        ("ragkit-mcp", "ragkit_mcp", {5}),
        ("ragkit-bench", "ragkit_bench", {4}),
    ):
        checks.append(
            Check(f"앱 {dist}", has_modules(module), version_of(dist), SYNC, labs)
        )
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    checks.append(
        Check(
            "디스크 여유 5GB 이상",
            free_gb >= 5,
            f"{free_gb:,.0f} GB 남음",
            "모델 · 인덱스 · 데이터에 약 3GB가 든다. 다른 파일을 정리한다",
            {0, 1, 2, 3, 4, 5},
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
            "Node.js 20+ 설치 후: corepack enable   (또는 brew install pnpm)",
            {5},
        )
    )
    return checks


def model_checks() -> list[Check]:
    checks = [
        Check(
            f"{rel(E5_DIR)} (학습 전 e5)",
            (E5_DIR / "config.json").exists(),
            size_mb(E5_DIR) if E5_DIR.exists() else "",
            GET_E5,
            {0, 1, 2, 3, 4, 5},
        ),
        Check(
            f"{rel(E5_DIR / 'onnx' / 'model.onnx')} (ONNX 추론용)",
            (E5_DIR / "onnx" / "model.onnx").exists(),
            fix=f"uv run ragkit export-onnx {rel(E5_DIR)}",
            labs={0, 2, 4, 5},
        ),
        Check(
            f"{rel(FINETUNED)} (파인튜닝, 실험 010 최고)",
            (FINETUNED / "model.safetensors").exists(),
            size_mb(FINETUNED) if FINETUNED.exists() else "",
            GET_FINETUNED.format(name=FINETUNED.name),
            {0, 3},
        ),
    ]
    for name in ("exp_002", "exp_004", "exp_006"):
        path = MODELS / "finetuned" / name
        checks.append(
            Check(
                f"{rel(path)} (실험 010 비교용)",
                (path / "model.safetensors").exists(),
                fix=GET_FINETUNED.format(name=name),
                labs={3},
                optional=True,
            )
        )
    int8 = MODELS / f"{E5_DIR.name}-int8"
    pruned = MODELS / f"{E5_DIR.name}-pruned-int8"
    checks += [
        Check(
            f"{rel(int8)} (INT8)",
            (int8 / "onnx" / "model.onnx").exists(),
            size_mb(int8) if int8.exists() else "",
            f"uv run ragkit quantize {rel(E5_DIR)}   (실습 4에서 직접 만든다)",
            {4},
        ),
        Check(
            f"{rel(pruned)} (가지치기 + INT8)",
            (pruned / "onnx" / "model.onnx").exists(),
            size_mb(pruned) if pruned.exists() else "",
            f"uv run ragkit prune-vocab {rel(E5_DIR)} → export-onnx → quantize"
            "   (실습 4에서 직접 만든다)",
            {4},
        ),
    ]
    return checks


def data_checks() -> list[Check]:
    corpus = DATA / "processed" / "law_docs.json"
    questions = sorted((DATA / dv.QUESTIONS_DIR).glob("*.jsonl"))
    checks = [
        Check(
            rel(corpus),
            corpus.exists(),
            size_mb(corpus) if corpus.exists() else "",
            GET_DATA + "   (직접 만들기: uv run python scripts/prepare_law_data.py)",
            {0, 1, 2, 3, 4, 5},
        ),
        Check(
            f"{rel(DATA / dv.QUESTIONS_DIR)}/ (스킬이 만든 질문)",
            bool(questions),
            f"파일 {len(questions)}개" if questions else "",
            GET_DATA
            + "   (또는 uv run python scripts/law_questions_drive.py download)",
            {1},
        ),
    ]
    for name in ("train", "dev", "test"):
        path = DATA / "splits" / f"{name}.jsonl"
        checks.append(
            Check(
                rel(path),
                path.exists(),
                f"질문 {jsonl_rows(path):,}개" if path.exists() else "",
                GET_DATA,
                {1, 2, 3},
            )
        )
    meta = DATA / "splits" / "split_meta.json"
    checks.append(Check(rel(meta), meta.exists(), fix=GET_DATA, labs={1, 3}))

    manifest_path = dv.versions_dir(DATA) / f"{DATA_VERSION}.json"
    if manifest_path.exists():
        diffs = dv.local_diff(dv.read_manifest(manifest_path), DATA)
        detail = "일치" if not diffs else f"다름 {len(diffs)}건 (예: {diffs[0]})"
        checks.append(
            Check(
                f"데이터 버전 {DATA_VERSION} 과 같은가",
                not diffs,
                detail,
                GET_DATA + " --force   (내가 만든 질문·분할이면 그대로 둬도 된다)",
                {1, 2, 3},
                optional=True,
            )
        )

    cache = default_cache_path(settings.gemini_model_name)
    checks.append(
        Check(
            f"{rel(cache)} (쿼리 확장 캐시)",
            cache.exists(),
            f"{jsonl_rows(cache):,}줄" if cache.exists() else "",
            FROM_DRIVE.format(path=rel(cache.parent)),
            {2},
        )
    )
    return checks


def index_checks() -> list[Check]:
    targets = [
        (
            model_key(E5),
            "학습 전 e5",
            f"uv run ragkit index   # {E5}",
            {0, 4, 5},
            f"uv run python scripts/finetuned_drive.py download {model_key(E5)}",
        ),
        (
            model_key(E5, FINETUNED) if FINETUNED.exists() else "r001_A-<학습 시각>",
            "파인튜닝 r001_A",
            f"uv run ragkit index --checkpoint {rel(FINETUNED)} --backend torch",
            {0},
            GET_FINETUNED.format(name=FINETUNED.name),
        ),
        (
            f"{E5_DIR.name}-int8",
            "INT8",
            f"uv run ragkit index --model {rel(MODELS / (E5_DIR.name + '-int8'))}",
            {4},
            None,
        ),
        (
            f"{E5_DIR.name}-pruned-int8",
            "가지치기 + INT8",
            f"uv run ragkit index --model {rel(MODELS / (E5_DIR.name + '-pruned-int8'))}",
            {4},
            None,
        ),
    ]
    checks = []
    for key, label, build, labs, get in targets:
        path = default_index_path(key)
        fix = (
            get or FROM_DRIVE.format(path=rel(path.parent))
        ) + f"\n        직접 만들기(수 분~십여 분): {build}"
        detail = size_mb(path) if path.exists() else ""
        checks.append(Check(f"{rel(path)} ({label})", path.exists(), detail, fix, labs))
    return checks


def key_checks() -> list[Check]:
    key = settings.gemini_api_key.strip()
    ok = key not in PLACEHOLDER_KEYS and not key.startswith("your_")
    if ok:
        detail = "설정됨 (여기서는 호출하지 않음)"
    elif key:
        detail = "예시 값 그대로"
    else:
        detail = "비어 있음"
    return [
        Check(
            "GEMINI_API_KEY (.env)",
            ok,
            detail,
            "cp .env.example .env 후 GEMINI_API_KEY에 AI Studio 키를 넣는다"
            "\n        (없으면 LLM 답변·쿼리 확장 단계만 건너뛴다)",
            {0, 2, 5},
            optional=True,
        )
    ]


# ============================================================
# 출력
# ============================================================
def mark(check: Check) -> str:
    if check.ok:
        return "✓"
    return "△" if check.optional else "✗"


def show_group(title: str, checks: list[Check]) -> None:
    print(f"\n[{title}]")
    name_w = max(width(c.name) for c in checks)
    detail_w = max(width(c.detail) for c in checks)
    for c in checks:
        labs = ",".join(str(n) for n in sorted(c.labs))
        detail = pad(c.detail, detail_w)
        extra = "  선택" if c.optional else ""
        print(f"  {mark(c)} {pad(c.name, name_w)}  {detail}  실습 {labs}{extra}")


def smoke_checks() -> list[Check]:
    """받은 모델과 인덱스를 실제로 열어 검색해 본다 (파일이 있어도 깨졌거나 짝이 안 맞을 수 있다)."""
    from ragkit.service import Searcher

    corpus = DATA / "processed" / "law_docs.json"
    n_docs = (
        len(json.loads(corpus.read_text(encoding="utf-8"))) if corpus.exists() else None
    )
    targets = [("학습 전 e5", {}, {0, 1, 2, 3, 4, 5})]
    if (FINETUNED / "model.safetensors").exists():
        targets.append(("파인튜닝 r001_A", {"checkpoint": FINETUNED}, {0, 3}))
    checks = []
    for label, kwargs, labs in targets:
        name = f"{label}: 열고 검색하기"
        start = time.perf_counter()
        try:
            searcher = Searcher.open(**kwargs)
            hits = searcher.search("야간 근로 수당", k=3)
        except Exception as e:  # noqa: BLE001 — 어떤 이유든 그대로 보여 준다
            checks.append(
                Check(
                    name,
                    False,
                    f"{type(e).__name__}: {str(e).splitlines()[0][:60]}",
                    "위 모델 · 인덱스 항목을 먼저 채운다",
                    labs,
                )
            )
            continue
        took = time.perf_counter() - start
        count = searcher.doc_count
        ok = bool(hits) and (n_docs is None or count == n_docs)
        detail = f"문서 {count:,}개 · {took:.1f}초"
        fix = "인덱스를 다시 받는다 (위 인덱스 항목의 명령)"
        if n_docs is not None and count != n_docs:
            detail += f" · 코퍼스 {n_docs:,}개와 다름"
            fix = "코퍼스와 인덱스 버전이 다르다. 둘 다 다시 받는다"
        checks.append(Check(name, ok, detail, fix, labs))
    return checks


def collect(quick: bool) -> dict[str, list[Check]]:
    groups = {
        "설치": install_checks(),
        "모델": model_checks(),
        "데이터": data_checks(),
        "인덱스": index_checks(),
        "API 키": key_checks(),
    }
    if not quick:
        print("동작 확인 중: 모델과 인덱스를 열어 검색해 봅니다 (수십 초)...")
        groups["동작 확인"] = smoke_checks()
    return groups


# --fix가 명령을 돌리는 순서: 설치 → 데이터 → 학습 전 모델 → ONNX → 파인튜닝 모델 · 인덱스
FIX_ORDER = (
    "uv sync",
    "uv run python scripts/data_version.py",
    "uv run python scripts/download_model_hf.py",
    "uv run ragkit export-onnx",
    "uv run python scripts/finetuned_drive.py",
)


def run_fixes(missing: list[Check]) -> None:
    cmds = sorted(
        {c.command() for c in missing if c.command()},
        key=lambda cmd: next(
            (i for i, p in enumerate(FIX_ORDER) if cmd.startswith(p)), len(FIX_ORDER)
        ),
    )
    if not cmds:
        print("--fix로 받을 수 있는 것이 없습니다. 아래 안내를 따라 손으로 채우세요.")
        return
    print("\n" + "=" * 60)
    print("--fix: 받을 수 있는 것을 받습니다")
    print("=" * 60)
    for cmd in cmds:
        print(f"\n$ {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(
                f"  실패 (종료 코드 {result.returncode}). 이어서 다음 명령을 실행합니다."
            )


def report(groups: dict[str, list[Check]]) -> list[Check]:
    print("=" * 60)
    print("실습 0-1: doctor — 환경과 리소스 점검")
    print("=" * 60)
    print(f"저장소: {ROOT}")
    print("✓ 있음   ✗ 없음(필요)   △ 없음(선택: 일부 단계만 건너뜀)")
    for title, checks in groups.items():
        show_group(title, checks)

    all_checks = [c for checks in groups.values() for c in checks]

    print("\n" + "=" * 60)
    print("실습별 준비 상태")
    print("=" * 60)
    LABS = {
        0: "환경·완성품·학습 전후 비교",
        1: "코퍼스·질문·분할",
        2: "평가·쿼리 확장",
        3: "학습·실험 비교",
        4: "ONNX·INT8·가지치기",
        5: "CLI·MCP·웹·MLflow",
    }
    for lab, title in LABS.items():
        need = [c for c in all_checks if lab in c.labs]
        missing = [c for c in need if not c.ok and not c.optional]
        skipped = [c for c in need if not c.ok and c.optional]
        status = "준비됨" if not missing else f"빠진 것 {len(missing)}개"
        print(f"  {'✓' if not missing else '✗'} 실습 {lab} ({title}): {status}")
        for c in missing:
            print(f"      ✗ {c.name}")
        for c in skipped:
            print(f"      △ {c.name}")

    missing = [c for c in all_checks if not c.ok]
    print("\n" + "=" * 60)
    print("빠진 것 받는 법")
    print("=" * 60)
    if not missing:
        print(
            "모두 준비됐습니다. 다음: uv run python lecture/03_setup/02_try_product.py"
        )
    else:
        # 같은 명령으로 받는 항목은 한데 모은다 (예: data_version.py pull 하나로 코퍼스·질문·분할)
        by_fix: dict[str, list[Check]] = {}
        for c in missing:
            by_fix.setdefault(c.fix, []).append(c)
        for fix, items in by_fix.items():
            for c in items:
                print(f"  {mark(c)} {c.name}")
            print(f"      → {fix}\n")
        print("받은 뒤 다시 실행한다. uv로 시작하는 명령은 --fix가 대신 돌린다.")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="강의 환경과 리소스를 점검한다")
    parser.add_argument(
        "--quick", action="store_true", help="파일과 패키지만 본다 (동작 확인 생략)"
    )
    parser.add_argument(
        "--fix", action="store_true", help="받을 수 있는 것을 받고 다시 점검한다"
    )
    args = parser.parse_args()

    missing = report(collect(args.quick))
    if args.fix and missing:
        run_fixes(missing)
        print("\n다시 점검합니다.")
        missing = report(collect(args.quick))
    sys.exit(1 if any(not c.optional for c in missing) else 0)


if __name__ == "__main__":
    main()
