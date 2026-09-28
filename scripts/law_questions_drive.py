"""법령 질문 데이터를 모아 zip으로 묶고, Google Drive에 올리거나 내려받는다.

사용법:
    uv run python scripts/law_questions_drive.py collect    # 스킬 테스트 질문 → data/questions_test/
    uv run python scripts/law_questions_drive.py bundle     # 질문 + 코퍼스 → dist/law-questions.zip
    uv run python scripts/law_questions_drive.py upload     # dist/의 zip을 Drive 폴더에 올림 (rclone)
    uv run python scripts/law_questions_drive.py download   # Drive 폴더에서 받아 data/law-questions/에 풂

업로드는 rclone을 쓴다. 처음 한 번 설치하고 로그인해 둔다:
    brew install rclone
    rclone config   # n → 이름 gdrive → Storage: drive → 나머지는 기본값, 브라우저에서 로그인

다운로드는 로그인이 필요 없다. Drive 폴더가 "링크가 있는 모든 사용자"로 공유되어 있어야 한다.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import gdown
from cyclopts import App

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_model_gdrive import extract_zip, sha256sum  # noqa: E402
from src.config import get_settings  # noqa: E402

# 강사의 Drive 폴더 ("링크가 있는 모든 사용자" 공유)
DEFAULT_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1y30fzc21thrATVzwBa9vzBWbOSTJt-U1"
)
BUNDLE_NAME = "law-questions"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT / ".claude" / "skills" / "law-question-gen-workspace"

# 스킬 테스트 반복 차수별로 질문이 가리키는 코퍼스.
# iteration-1은 긴 조문을 나누기 전(조 단위 id), iteration-2부터는 항·호 조각 id다.
ITERATION_CORPUS = {1: "law_docs_article_level.json", 2: "law_docs.json"}
# 원본 레코드에서 그대로 옮기는 선택 필드. 없으면 null(리스트는 빈 리스트)로 채운다.
OPTIONAL_FIELDS = {"query_type": None, "answer": None, "related_ids": []}

app = App(help="법령 질문 데이터를 묶어 Google Drive에 올리고 내려받는다.")


def folder_id(url: str) -> str:
    """Drive 폴더 링크에서 폴더 ID를 꺼낸다."""
    return url.rstrip("/").split("/folders/")[-1].split("?")[0]


def normalize(record: dict, *, iteration: int, eval_name: str, config: str) -> dict:
    """테스트 실행마다 조금씩 다른 레코드를 공통 형식으로 맞춘다."""
    normalized = {
        "query": record["query"],
        "positive_id": record["positive_id"],
        "hard_negative_ids": record.get("hard_negative_ids", []),
    }
    for field, default in OPTIONAL_FIELDS.items():
        normalized[field] = record.get(field, default)
    normalized["corpus"] = ITERATION_CORPUS[iteration]
    normalized["source"] = {"iteration": iteration, "eval": eval_name, "config": config}
    return normalized


def collect_records(workspace: Path) -> list[dict]:
    """스킬 워크스페이스의 모든 테스트 질문을 공통 형식으로 모은다.

    query, positive_id, hard_negative_ids가 모두 있는 파일만 쓴다
    (baseline이 따로 만든 triplets.jsonl은 queries.jsonl에서 파생된 것이라 형식이 달라 빠진다. 원본은 raw/에 남는다).
    """
    records: list[dict] = []
    for path in sorted(workspace.glob("iteration-*/eval-*/*/run-*/outputs/*.jsonl")):
        run_dir = path.parents[1]
        iteration = int(run_dir.parents[2].name.split("-")[1])
        eval_name = run_dir.parents[1].name.split("-", 2)[2]
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if not rows or not {"query", "positive_id", "hard_negative_ids"} <= rows[0].keys():
            continue
        records += [
            normalize(row, iteration=iteration, eval_name=eval_name, config=run_dir.parent.name)
            for row in rows
        ]
    return records


@app.command
def collect(workspace: Path = WORKSPACE, output: Path | None = None) -> None:
    """스킬 테스트(iteration-1·2, with_skill·without_skill) 질문을 한 폴더에 모은다.

    Args:
        workspace: law-question-gen 스킬 워크스페이스.
        output: 저장할 폴더. 기본값은 data/questions_test.
    """
    output = output or get_settings().data_dir / "questions_test"
    records = collect_records(workspace)
    if not records:
        raise SystemExit(f"{workspace}에서 질문을 찾지 못했습니다.")

    raw_dir = output / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    for path in workspace.glob("iteration-*/eval-*/*/run-*/outputs/*.jsonl"):
        target = raw_dir / path.relative_to(workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "questions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[tuple, int] = {}
    for r in records:
        key = (r["source"]["iteration"], r["source"]["eval"], r["source"]["config"])
        counts[key] = counts.get(key, 0) + 1
    for (iteration, eval_name, config), n in sorted(counts.items()):
        print(f"  iteration-{iteration} {eval_name:22s} {config:14s} {n:4d}")
    print(f"완료: 질문 {len(records)}개 → {output / 'questions.jsonl'}")


@app.command
def bundle(
    questions_dir: Path | None = None,
    corpus_dir: Path | None = None,
    output: Path | None = None,
) -> None:
    """질문 폴더와 코퍼스를 zip 하나로 묶고 sha256 파일을 만든다.

    zip 구조: law-questions/{README.md, questions/, corpus/}

    Args:
        questions_dir: collect로 만든 폴더. 기본값은 data/questions_test.
        corpus_dir: 코퍼스 JSON이 있는 폴더. 기본값은 data/processed.
        output: zip 경로. 기본값은 dist/law-questions.zip.
    """
    settings = get_settings()
    questions_dir = questions_dir or settings.data_dir / "questions_test"
    corpus_dir = corpus_dir or settings.data_dir / "processed"
    output = output or PROJECT_ROOT / "dist" / f"{BUNDLE_NAME}.zip"

    corpora = [corpus_dir / name for name in sorted(set(ITERATION_CORPUS.values()))]
    if missing := [p for p in [questions_dir / "questions.jsonl", *corpora] if not p.exists()]:
        raise SystemExit(f"파일이 없습니다: {[str(p) for p in missing]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(questions_dir.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(questions_dir)
                if path.name == "README.md":
                    zf.write(path, f"{BUNDLE_NAME}/README.md")
                else:
                    zf.write(path, f"{BUNDLE_NAME}/questions/{arcname}")
        for path in corpora:
            zf.write(path, f"{BUNDLE_NAME}/corpus/{path.name}")

    digest = sha256sum(output)
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"완료: {output} ({output.stat().st_size / 1e6:.1f}MB, sha256 {digest[:12]}…)")


@app.command
def upload(
    folder_url: str = DEFAULT_FOLDER_URL,
    remote: str = "gdrive",
    zip_path: Path | None = None,
) -> None:
    """bundle로 만든 zip과 sha256 파일을 Drive 폴더에 올린다 (같은 이름이면 덮어쓴다).

    Args:
        folder_url: 올릴 Drive 폴더 링크.
        remote: rclone config에서 만든 Google Drive remote 이름.
        zip_path: 올릴 zip. 기본값은 dist/law-questions.zip.
    """
    zip_path = zip_path or PROJECT_ROOT / "dist" / f"{BUNDLE_NAME}.zip"
    files = [zip_path, zip_path.with_suffix(".zip.sha256")]
    if missing := [p for p in files if not p.exists()]:
        raise SystemExit(f"먼저 bundle을 실행하세요. 없는 파일: {[str(p) for p in missing]}")
    if shutil.which("rclone") is None:
        raise SystemExit("rclone이 없습니다. `brew install rclone` 후 `rclone config`로 로그인하세요.")

    for path in files:
        print(f"업로드 중: {path.name} → {remote}: (폴더 {folder_id(folder_url)})")
        subprocess.run(
            ["rclone", "copyto", str(path), f"{remote}:{path.name}",
             f"--drive-root-folder-id={folder_id(folder_url)}", "--progress"],
            check=True,
        )
    print("완료")


@app.command
def download(
    folder_url: str = DEFAULT_FOLDER_URL,
    output_dir: Path | None = None,
    force: bool = False,
) -> None:
    """Drive 폴더에서 zip을 받아 sha256을 확인하고 data/law-questions/에 푼다.

    Args:
        folder_url: Drive 폴더 링크 ("링크가 있는 모든 사용자" 공유).
        output_dir: 풀 폴더. 기본값은 data/law-questions.
        force: 이미 받은 데이터가 있어도 다시 받는다.
    """
    output_dir = output_dir or get_settings().data_dir / BUNDLE_NAME
    if (output_dir / "questions" / "questions.jsonl").exists() and not force:
        print(f"이미 존재합니다: {output_dir} (다시 받으려면 --force)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        files = gdown.download_folder(url=folder_url, output=tmp, quiet=False)
        if not files:
            raise SystemExit("다운로드 실패. 폴더 공유가 '링크가 있는 모든 사용자'인지 확인하세요.")
        zip_path = Path(tmp) / f"{BUNDLE_NAME}.zip"
        sha_path = Path(tmp) / f"{BUNDLE_NAME}.zip.sha256"
        if not zip_path.exists():
            raise SystemExit(f"폴더에 {zip_path.name}이 없습니다. 받은 파일: {files}")
        if sha_path.exists():
            expected = sha_path.read_text(encoding="utf-8").split()[0]
            if (actual := sha256sum(zip_path)) != expected:
                raise SystemExit(f"sha256 불일치 (기대: {expected}, 실제: {actual}).")
            print("sha256 확인 완료")
        extract_zip(zip_path, output_dir)
    print(f"완료: {output_dir}")


if __name__ == "__main__":
    app()
