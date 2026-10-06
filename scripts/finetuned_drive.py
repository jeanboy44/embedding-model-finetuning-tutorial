"""파인튜닝한 모델(과 그 인덱스)을 Google Drive에 올리고 내려받는다.

모델 하나 = zip 하나. zip에는 models/finetuned/<이름>/ 과, 있으면 그 모델로 만든 인덱스
data/processed/index/<모델 키>.sqlite 가 저장소 루트 기준 경로로 들어간다.
올린 결과(파일 id, sha256, 크기, 가중치 수정 시각)는 scripts/finetuned_models.json 에 적고 커밋한다.

사용법:
    uv run python scripts/finetuned_drive.py list                     # 받을 수 있는 모델
    uv run python scripts/finetuned_drive.py download                 # r001_A + 학습 전 e5 인덱스 (강의 기본)
    uv run python scripts/finetuned_drive.py download exp_002 exp_004 exp_006   # 6교시 비교용
    uv run python scripts/finetuned_drive.py download --all
    uv run python scripts/finetuned_drive.py upload r001_A exp_002   # (강사) zip → Drive, 목록 갱신
    uv run python scripts/finetuned_drive.py upload-index multilingual-e5-small   # (강사) 인덱스만

업로드는 rclone을 쓴다 (scripts/data_version.py와 같은 remote):
    brew install rclone
    rclone config   # n → 이름 gdrive → Storage: drive → 나머지는 기본값, 브라우저에서 로그인

다운로드는 로그인이 필요 없다. Drive 폴더가 "링크가 있는 모든 사용자"로 공유되어 있어야 한다.

인덱스 파일 이름의 모델 키는 가중치 파일의 수정 시각으로 정해진다(ragkit.retrieval.model_key).
그래서 받은 뒤 model.safetensors의 수정 시각을 올릴 때 값으로 되돌려, 받은 인덱스를 그대로 찾게 한다.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import gdown
from cyclopts import App

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.config import get_settings
from ragkit.retrieval import model_key
from scripts.download_model_gdrive import sha256sum

# 강사의 Drive 폴더 ("링크가 있는 모든 사용자" 공유)
DEFAULT_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1vpGtkFmS5kPfRlu5f7pcGKf_gZ2AWHXK"
)
MANIFEST = Path(__file__).resolve().parent / "finetuned_models.json"
DEFAULT_MODEL = "r001_A"  # 실험 010에서 test R@5가 가장 높은 모델 (3교시 · 실습 0)
DEFAULT_INDEX = "multilingual-e5-small"  # 학습 전 e5의 인덱스. 직접 만들면 7~13분
WEIGHTS = "model.safetensors"

app = App(help="파인튜닝한 모델과 인덱스를 Google Drive에 올리고 내려받는다.")


def folder_id(url: str) -> str:
    """Drive 폴더 링크에서 폴더 ID를 꺼낸다."""
    return url.rstrip("/").split("/folders/")[-1].split("?")[0]


def load_manifest(path: Path = MANIFEST) -> dict:
    """올린 모델 목록. 없으면 빈 목록."""
    if not path.exists():
        return {"drive_folder": DEFAULT_FOLDER_URL, "models": {}, "indexes": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("indexes", {})
    return manifest


def save_manifest(manifest: dict, path: Path = MANIFEST) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def bundle(name: str, root: Path, output: Path, *, with_index: bool = True) -> dict:
    """models/finetuned/<name>/ (+ 인덱스)를 zip 하나로 묶고, 목록에 적을 정보를 돌려준다.

    zip 안의 경로는 root 기준(models/finetuned/<name>/..., data/processed/index/...)이라
    받는 쪽에서 저장소 루트에 그대로 풀면 된다.
    """
    model_dir = root / "models" / "finetuned" / name
    weights = model_dir / WEIGHTS
    if not weights.exists():
        raise SystemExit(f"모델이 없습니다: {weights}")

    key = model_key(name, model_dir)
    index = root / "data" / "processed" / "index" / f"{key}.sqlite"
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    if with_index and index.exists():
        files.append(index)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(root).as_posix())

    return {
        "file": output.name,
        "model_dir": f"models/finetuned/{name}",
        "index": index.relative_to(root).as_posix()
        if with_index and index.exists()
        else None,
        "model_key": key,
        "weights_mtime": weights.stat().st_mtime,
        "sha256": sha256sum(output),
        "size_mb": round(output.stat().st_size / 1e6, 1),
    }


def bundle_index(key: str, root: Path, output: Path) -> dict:
    """인덱스 파일 하나(data/processed/index/<key>.sqlite)를 zip으로 묶는다."""
    index = root / "data" / "processed" / "index" / f"{key}.sqlite"
    if not index.exists():
        raise SystemExit(f"인덱스가 없습니다: {index}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(index, index.relative_to(root).as_posix())
    return {
        "file": output.name,
        "index": index.relative_to(root).as_posix(),
        "sha256": sha256sum(output),
        "size_mb": round(output.stat().st_size / 1e6, 1),
    }


def install(zip_path: Path, entry: dict, root: Path) -> None:
    """zip을 저장소 루트에 풀고, 가중치 수정 시각을 올릴 때 값으로 되돌린다."""
    root = root.resolve()
    model_dir = root / entry["model_dir"] if entry.get("model_dir") else None
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"안전하지 않은 zip 경로: {member}")
        if model_dir is not None and model_dir.exists():
            shutil.rmtree(model_dir)
        zf.extractall(root)
    if model_dir is not None:
        mtime = entry["weights_mtime"]
        os.utime(model_dir / WEIGHTS, (mtime, mtime))


def _drive_file(remote: str, folder: str, name: str) -> dict | None:
    """Drive 폴더에서 이름이 같은 파일 정보(rclone lsjson 한 줄). 없으면 None."""
    out = subprocess.run(
        [
            "rclone",
            "lsjson",
            f"{remote}:",
            f"--drive-root-folder-id={folder}",
            "--files-only",
            "--include",
            name,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = json.loads(out or "[]")
    return rows[0] if rows else None


def _upload_zip(
    zip_path: Path, remote: str, folder: str, *, transfer: bool = True
) -> str:
    """zip 하나를 Drive 폴더에 올리고 파일 ID를 돌려준다.

    transfer=False면 올리지 않고, 브라우저로 직접 올려 둔 같은 이름의 파일 ID만 찾는다.
    """
    if not transfer:
        found = _drive_file(remote, folder, zip_path.name)
        if found is None:
            raise SystemExit(
                f"Drive 폴더에 {zip_path.name}이 없습니다. 먼저 브라우저로 올리세요."
            )
        print(f"  Drive에 있는 파일 사용: file_id={found['ID']}")
        return found["ID"]
    print(f"업로드 중: {zip_path.name} → {remote}: (폴더 {folder})")
    subprocess.run(
        [
            "rclone",
            "copyto",
            str(zip_path),
            f"{remote}:{zip_path.name}",
            f"--drive-root-folder-id={folder}",
            "--stats=30s",
            "--stats-one-line",
            "-v",
        ],
        check=True,
    )
    uploaded = _drive_file(remote, folder, zip_path.name)
    if uploaded is None:
        raise SystemExit(f"업로드한 파일을 Drive에서 찾지 못했습니다: {zip_path.name}")
    print(f"  완료: file_id={uploaded['ID']}")
    return uploaded["ID"]


def _require_rclone() -> None:
    if shutil.which("rclone") is None:
        raise SystemExit(
            "rclone이 없습니다. `brew install rclone` 후 `rclone config`로 로그인하세요."
        )


@app.command
def upload(
    *names: str,
    folder_url: str = DEFAULT_FOLDER_URL,
    remote: str = "gdrive",
    with_index: bool = True,
    transfer: bool = True,
) -> None:
    """(강사) 모델을 zip으로 묶어 Drive 폴더에 올리고 scripts/finetuned_models.json을 갱신한다.

    Args:
        names: models/finetuned/ 아래 폴더 이름. 비우면 r001_A.
        folder_url: 올릴 Drive 폴더 링크 ("링크가 있는 모든 사용자" 공유).
        remote: rclone config에서 만든 Google Drive remote 이름.
        with_index: 그 모델로 만든 인덱스(data/processed/index/<모델 키>.sqlite)도 함께 묶는다.
        transfer: False(--no-transfer)면 zip만 만들고, 브라우저로 올려 둔 파일의 ID를 찾아 적는다.
            rclone이 요청 한도에 걸려 업로드가 멈출 때 쓴다.
    """
    _require_rclone()
    root = get_settings().project_root
    folder = folder_id(folder_url)
    for name in names or (DEFAULT_MODEL,):
        zip_path = root / "dist" / f"finetuned-{name}.zip"
        print(f"묶는 중: models/finetuned/{name} → {zip_path.relative_to(root)}")
        entry = bundle(name, root, zip_path, with_index=with_index)
        print(f"  {entry['size_mb']} MB, 인덱스 {'포함' if entry['index'] else '없음'}")
        entry["file_id"] = _upload_zip(zip_path, remote, folder, transfer=transfer)
        manifest = load_manifest()
        manifest["drive_folder"] = folder_url
        manifest["models"][name] = entry
        save_manifest(manifest)
    print(f"\n목록 갱신: {MANIFEST.relative_to(MANIFEST.parents[1])} (커밋하세요)")


@app.command(name="upload-index")
def upload_index(
    *keys: str,
    folder_url: str = DEFAULT_FOLDER_URL,
    remote: str = "gdrive",
    transfer: bool = True,
) -> None:
    """(강사) 인덱스 파일만 zip으로 묶어 올린다. 학습 전 모델의 인덱스처럼 모델은 따로 받는 경우.

    Args:
        keys: 인덱스 이름(data/processed/index/<key>.sqlite). 비우면 multilingual-e5-small.
        folder_url: 올릴 Drive 폴더 링크.
        remote: rclone remote 이름.
        transfer: False(--no-transfer)면 zip만 만들고, 브라우저로 올려 둔 파일의 ID를 찾아 적는다.
    """
    _require_rclone()
    root = get_settings().project_root
    folder = folder_id(folder_url)
    for key in keys or (DEFAULT_INDEX,):
        zip_path = root / "dist" / f"index-{key}.zip"
        print(
            f"묶는 중: data/processed/index/{key}.sqlite → {zip_path.relative_to(root)}"
        )
        entry = bundle_index(key, root, zip_path)
        entry["file_id"] = _upload_zip(zip_path, remote, folder, transfer=transfer)
        manifest = load_manifest()
        manifest["indexes"][key] = entry
        save_manifest(manifest)
    print(f"\n목록 갱신: {MANIFEST.relative_to(MANIFEST.parents[1])} (커밋하세요)")


def _fetch(entry: dict, root: Path) -> None:
    """zip 하나를 받아 sha256을 확인하고 저장소에 푼다."""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / entry["file"]
        print(f"다운로드 중: {entry['file']} ({entry['size_mb']} MB)")
        if gdown.download(id=entry["file_id"], output=str(zip_path)) is None:
            raise SystemExit(
                "다운로드 실패. Drive 폴더 공유가 '링크가 있는 모든 사용자'인지 확인하세요."
            )
        if sha256sum(zip_path) != entry["sha256"]:
            raise SystemExit(
                f"sha256이 맞지 않습니다: {entry['file']} (다시 받아 보세요)"
            )
        install(zip_path, entry, root)


@app.command
def download(*names: str, all: bool = False, force: bool = False) -> None:
    """Drive에서 모델(과 그 인덱스)을 받아 저장소에 둔다. 로그인 불필요.

    이름 없이 부르면 강의 기본 묶음(r001_A + 학습 전 e5 인덱스)을 받는다.

    Args:
        names: 받을 모델 이름(models/finetuned/ 아래). 인덱스 이름도 된다.
        all: 목록에 있는 모델과 인덱스를 모두 받는다.
        force: 이미 있어도 다시 받는다.
    """
    manifest = load_manifest()
    models, indexes = manifest["models"], manifest["indexes"]
    if not models and not indexes:
        raise SystemExit(
            f"{MANIFEST.name}에 올린 것이 없습니다. 강사가 먼저 upload 해야 합니다."
        )
    if all:
        targets = [*models, *indexes]
    elif names:
        targets = list(names)
    else:
        targets = [
            n for n in (DEFAULT_MODEL, DEFAULT_INDEX) if n in models or n in indexes
        ]
    if unknown := [n for n in targets if n not in models and n not in indexes]:
        raise SystemExit(
            f"목록에 없는 이름: {', '.join(unknown)} (있는 것: {', '.join([*models, *indexes])})"
        )

    root = get_settings().project_root
    for name in targets:
        entry = models.get(name) or indexes[name]
        model_dir = root / entry["model_dir"] if entry.get("model_dir") else None
        index = root / entry["index"] if entry.get("index") else None
        have_model = model_dir is None or (model_dir / WEIGHTS).exists()
        have_index = index is None or index.exists()
        if have_model and have_index and not force:
            print(f"이미 있습니다: {name} (다시 받으려면 --force)")
            continue
        _fetch(entry, root)
        if model_dir is not None:
            print(f"  모델: {entry['model_dir']}")
        if index is not None:
            print(f"  인덱스: {entry['index']}")
        if (
            model_dir is not None
            and index is not None
            and model_key(name, model_dir) != entry["model_key"]
        ):
            print(
                "  경고: 모델 키가 인덱스 이름과 다릅니다. --index로 인덱스를 직접 지정하세요."
            )


@app.command(name="list")
def list_models() -> None:
    """Drive에 올라간 모델 · 인덱스 목록과 로컬에 받았는지 보여 준다."""
    manifest = load_manifest()
    root = get_settings().project_root
    print(f"Drive 폴더: {manifest['drive_folder']}")
    for name, entry in manifest["models"].items():
        have = (root / entry["model_dir"] / WEIGHTS).exists()
        index = "인덱스 포함" if entry.get("index") else "인덱스 없음"
        print(
            f"  모델   {name:22s} {entry['size_mb']:7.1f} MB  {index}  [{'받음' if have else '-'}]"
        )
    for key, entry in manifest["indexes"].items():
        have = (root / entry["index"]).exists()
        print(
            f"  인덱스 {key:22s} {entry['size_mb']:7.1f} MB  [{'받음' if have else '-'}]"
        )


if __name__ == "__main__":
    app()
