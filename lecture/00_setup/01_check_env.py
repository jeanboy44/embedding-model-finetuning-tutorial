"""
실습 0-1 (3교시): 환경 점검
===========================

학습 목표:
- 강의에 필요한 설치(ragkit, extra, 앱, pnpm) · 모델 · 데이터 · 인덱스 · API 키를 한 번에 점검한다
- 빠진 것마다 어떤 명령으로 받거나 만드는지 확인한다
- 실습 0~5 중 지금 바로 할 수 있는 실습이 어디까지인지 본다

이 스크립트는 파일과 패키지가 있는지만 본다. 모델을 불러오거나 Gemini를 호출하지 않으므로
몇 초 안에 끝난다. 확인만 하는 스크립트라 `--run` 모드는 없다.

사전 준비:
    uv sync --all-packages --all-extras        # ragkit + 모든 앱 + 학습 도구

실행:
    uv run python lecture/00_setup/01_check_env.py
"""

import importlib.metadata
import importlib.util
import shutil
import sys
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
            FROM_DRIVE.format(path=rel(FINETUNED)),
            {0, 3},
        ),
    ]
    for name in ("exp_002", "exp_004", "exp_006"):
        path = MODELS / "finetuned" / name
        checks.append(
            Check(
                f"{rel(path)} (실험 010 비교용)",
                (path / "model.safetensors").exists(),
                fix=FROM_DRIVE.format(path=rel(path)),
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
        (model_key(E5), "학습 전 e5", f"uv run ragkit index   # {E5}", {0, 4, 5}),
        (
            model_key(E5, FINETUNED) if FINETUNED.exists() else "r001_A-<학습 시각>",
            "파인튜닝 r001_A",
            f"uv run ragkit index --checkpoint {rel(FINETUNED)} --backend torch",
            {0},
        ),
        (
            f"{E5_DIR.name}-int8",
            "INT8",
            f"uv run ragkit index --model {rel(MODELS / (E5_DIR.name + '-int8'))}",
            {4},
        ),
        (
            f"{E5_DIR.name}-pruned-int8",
            "가지치기 + INT8",
            f"uv run ragkit index --model {rel(MODELS / (E5_DIR.name + '-pruned-int8'))}",
            {4},
        ),
    ]
    checks = []
    for key, label, build, labs in targets:
        path = default_index_path(key)
        fix = (
            FROM_DRIVE.format(path=rel(path.parent))
            + f"\n        직접 만들기(수 분~십여 분): {build}"
        )
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


groups = {
    "설치": install_checks(),
    "모델": model_checks(),
    "데이터": data_checks(),
    "인덱스": index_checks(),
    "API 키": key_checks(),
}

print("=" * 60)
print("실습 0-1: 환경 점검")
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
    0: "환경·완성품·왜 RAG",
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
    print("모두 준비됐습니다. 다음: uv run python lecture/00_setup/02_try_product.py")
else:
    # 같은 명령으로 받는 항목은 한데 모은다 (예: data_version.py pull 하나로 코퍼스·질문·분할)
    by_fix: dict[str, list[Check]] = {}
    for c in missing:
        by_fix.setdefault(c.fix, []).append(c)
    for fix, items in by_fix.items():
        for c in items:
            print(f"  {mark(c)} {c.name}")
        print(f"      → {fix}\n")
    print("받은 뒤 이 스크립트를 다시 실행해 확인한다.")
