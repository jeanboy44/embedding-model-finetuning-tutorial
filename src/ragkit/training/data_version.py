"""데이터 버전: 학습·평가 데이터(코퍼스 + 분할 + 원본 질문)를 v1, v2, …로 고정한다.

실제 파일은 버전별 zip(law-data-v1.zip)으로 Google Drive에 두고, git에는 manifest
(data/versions/v1.json: 파일별 sha256·행 수·zip 위치)만 커밋한다. 한 번 만든 버전은 고치지 않고,
질문·코퍼스·분할이 바뀌면 새 버전을 만든다.

    manifest = build_manifest(data_dir, "v1", description="...")   # 지금 data/를 v1으로
    refs = match_files({"corpus": corpus_path, "train": train_path})  # 이 파일들이 어느 버전인가
    diffs = local_diff(manifest, data_dir)                            # 지금 data/와 v1의 차이

MLflow 기록(데이터셋 등록, run에 입력 연결)은 ragkit.tracking이 맡는다.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

DATASET_NAME = "law-retrieval"
# 역할 → data/ 아래 경로. zip 안에서도 같은 경로를 쓴다(law-data-v1/<경로>).
FILES = {
    "corpus": "processed/law_docs.json",
    "train": "splits/train.jsonl",
    "dev": "splits/dev.jsonl",
    "test": "splits/test.jsonl",
    "split_meta": "splits/split_meta.json",
}
QUESTIONS_DIR = "questions"  # 스킬로 만든 원본 질문(<법령>__pNN.jsonl). 분할을 다시 만들 때 쓴다.
DIGEST_LEN = 16  # MLflow 데이터셋 digest 칸은 36자까지


@dataclass(frozen=True)
class DatasetRef:
    """run 입력으로 남길 데이터셋 하나: 어느 버전의 어떤 파일인지."""

    name: str  # law-retrieval/train
    digest: str  # 파일 sha256 앞부분
    source: str  # 버전 zip을 받을 수 있는 주소 (Drive)
    version: str  # v1
    role: str  # train


def versions_dir(data_dir: Path) -> Path:
    return data_dir / "versions"


def zip_name(version: str) -> str:
    return f"law-data-{version}.zip"


def sha256sum(path: Path) -> str:
    """파일의 sha256 (16진수)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _cached_sha(path: str, size: int, mtime_ns: int) -> str:
    return sha256sum(Path(path))


def file_sha(path: Path) -> str:
    """같은 파일(경로·크기·수정 시각)이면 해시를 다시 계산하지 않는다 (train 한 번에 코퍼스를 여러 번 묻는다)."""
    stat = path.stat()
    return _cached_sha(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def count_rows(path: Path) -> int:
    """JSONL은 비어 있지 않은 줄 수, JSON 배열은 항목 수, 그 밖의 JSON은 0."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, list) else 0


def questions_sha(files: dict[str, str]) -> str:
    """질문 폴더 전체의 해시: 파일 이름순으로 '이름 해시' 줄을 이어 해시한다."""
    lines = "".join(f"{name} {sha}\n" for name, sha in sorted(files.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def build_manifest(data_dir: Path, version: str, *, description: str = "") -> dict:
    """지금 data/의 파일로 manifest를 만든다 (zip·Drive 정보는 나중에 채운다).

    Raises:
        FileNotFoundError: 버전에 넣을 파일이 없을 때.
    """
    missing = [rel for rel in FILES.values() if not (data_dir / rel).exists()]
    question_files = sorted((data_dir / QUESTIONS_DIR).glob("*.jsonl"))
    if not question_files:
        missing.append(f"{QUESTIONS_DIR}/*.jsonl")
    if missing:
        raise FileNotFoundError(f"버전에 넣을 파일이 없습니다: {missing}")

    files = {}
    for role, rel in FILES.items():
        path = data_dir / rel
        entry = {"path": rel, "sha256": sha256sum(path), "bytes": path.stat().st_size}
        if role != "split_meta":
            entry["rows"] = count_rows(path)
        files[role] = entry
    shas = {p.name: sha256sum(p) for p in question_files}
    files["questions"] = {
        "path": f"{QUESTIONS_DIR}/",
        "sha256": questions_sha(shas),
        "bytes": sum(p.stat().st_size for p in question_files),
        "rows": sum(count_rows(p) for p in question_files),
        "files": shas,
    }
    return {
        "name": DATASET_NAME,
        "version": version,
        "created": datetime.now().astimezone().date().isoformat(),
        "description": description,
        "files": files,
        "zip": {"name": zip_name(version), "sha256": None, "bytes": None, "drive_file_id": None},
    }


def source_url(manifest: dict) -> str:
    """버전 zip을 받을 주소. Drive에 올린 뒤면 파일 링크, 아니면 Drive 폴더 링크."""
    zip_info = manifest.get("zip") or {}
    if file_id := zip_info.get("drive_file_id"):
        return f"https://drive.google.com/uc?id={file_id}"
    return manifest.get("drive_folder") or "https://drive.google.com/"


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_manifests(directory: Path) -> list[dict]:
    """data/versions/의 manifest를 버전 번호순(v1, v2, …, v10)으로."""
    paths = sorted(directory.glob("v*.json"), key=lambda p: (len(p.stem), p.stem))
    return [read_manifest(p) for p in paths]


def to_ref(manifest: dict, role: str) -> DatasetRef:
    entry = manifest["files"][role]
    return DatasetRef(
        name=f"{manifest['name']}/{role}",
        digest=entry["sha256"][:DIGEST_LEN],
        source=source_url(manifest),
        version=manifest["version"],
        role=role,
    )


def match_files(paths: dict[str, Path], directory: Path) -> tuple[list[DatasetRef], list[str]]:
    """파일마다 내용(sha256)이 같은 버전 파일을 찾는다.

    Args:
        paths: run 입력 이름(context) → 파일. 예: {"corpus": law_docs.json, "test": test.jsonl}.
        directory: manifest 폴더 (data/versions).

    Returns:
        (찾은 데이터셋, 어느 버전에도 없는 입력 이름). 데이터셋 이름은 버전 쪽 역할을 따른다
        (compare가 data/splits/test.jsonl을 넘기면 law-retrieval/test).
    """
    by_sha: dict[str, tuple[dict, str]] = {}
    for manifest in load_manifests(directory):  # 같은 파일이 여러 버전에 있으면 먼저 만든 버전
        for role, entry in manifest["files"].items():
            if role != "questions":
                by_sha.setdefault(entry["sha256"], (manifest, role))
    found, unmatched = [], []
    for context, path in paths.items():
        hit = by_sha.get(file_sha(path)) if path.is_file() else None
        if hit:
            found.append(to_ref(*hit))
        else:
            unmatched.append(context)
    return found, unmatched


def version_tag(found: list[DatasetRef], unmatched: list[str]) -> str:
    """run에 붙일 data_version 태그: v1, 섞였으면 v1+v2, 버전 밖 파일이 있으면 +unversioned."""
    versions = sorted({ref.version for ref in found}, key=lambda v: (len(v), v))
    return "+".join(versions + (["unversioned"] if unmatched else [])) or "unversioned"


def local_diff(manifest: dict, data_dir: Path) -> list[str]:
    """지금 data/가 이 버전과 다른 점 (같으면 빈 목록)."""
    diffs = []
    for role, entry in manifest["files"].items():
        if role == "questions":
            continue
        path = data_dir / entry["path"]
        if not path.exists():
            diffs.append(f"없음: {entry['path']}")
        elif file_sha(path) != entry["sha256"]:
            diffs.append(f"다름: {entry['path']}")
    expected = manifest["files"]["questions"]["files"]
    local = {p.name: p for p in (data_dir / QUESTIONS_DIR).glob("*.jsonl")}
    diffs += [f"없음: {QUESTIONS_DIR}/{n}" for n in sorted(expected.keys() - local.keys())]
    diffs += [f"추가됨: {QUESTIONS_DIR}/{n}" for n in sorted(local.keys() - expected.keys())]
    diffs += [
        f"다름: {QUESTIONS_DIR}/{n}"
        for n in sorted(expected.keys() & local.keys())
        if file_sha(local[n]) != expected[n]
    ]
    return diffs
