"""데이터 버전 관리: 지금 data/를 버전(v1, v2, …)으로 고정해 Drive에 올리고 MLflow에 등록한다.

버전 = 코퍼스(processed/law_docs.json) + 분할(splits/) + 원본 질문(questions/).
실제 파일은 Drive의 law-data-<버전>.zip, git에는 manifest(data/versions/<버전>.json)만 커밋한다.
한 번 올린 버전은 고치지 않는다. 데이터가 바뀌면 새 버전을 만든다.

사용법:
    uv run python scripts/data_version.py create v1 --description "..."   # manifest + dist/law-data-v1.zip
    uv run python scripts/data_version.py upload v1      # zip을 Drive에 올리고 manifest에 파일 id를 적음 (rclone)
    uv run python scripts/data_version.py register v1    # MLflow law-data 실험에 data/v1 run (MLFLOW_TRACKING_URI)
    uv run python scripts/data_version.py pull v1        # Drive에서 받아 data/에 풂 (로그인 불필요)
    uv run python scripts/data_version.py status         # 지금 data/가 어느 버전과 같은지

이후 ragkit train·evaluate·compare는 입력 파일이 어느 버전인지 찾아 run에 데이터셋과 data_version 태그를 남긴다.
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
from ragkit import tracking
from ragkit.config import get_settings
from ragkit.training import data_version as dv
from scripts.download_model_gdrive import extract_zip
from scripts.law_questions_drive import DEFAULT_FOLDER_URL, folder_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = App(help="데이터 버전(코퍼스 + 분할 + 원본 질문)을 만들고 Drive·MLflow에 올린다.")


def manifest_path(version: str, data_dir: Path | None = None) -> Path:
    return dv.versions_dir(data_dir or get_settings().data_dir) / f"{version}.json"


def _load(version: str) -> tuple[dict, Path]:
    path = manifest_path(version)
    if not path.exists():
        raise SystemExit(f"manifest가 없습니다: {path}\n  먼저 실행: data_version.py create {version}")
    return dv.read_manifest(path), path


def write_zip(manifest: dict, data_dir: Path, output: Path) -> None:
    """manifest의 파일을 law-data-<버전>/<data 아래 경로>로 묶는다 (풀면 data/와 같은 구조)."""
    root = f"law-data-{manifest['version']}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for role, entry in manifest["files"].items():
            if role == "questions":
                for name in sorted(entry["files"]):
                    zf.write(data_dir / dv.QUESTIONS_DIR / name, f"{root}/{dv.QUESTIONS_DIR}/{name}")
            else:
                zf.write(data_dir / entry["path"], f"{root}/{entry['path']}")


@app.command
def create(version: str, *, description: str = "", force: bool = False) -> None:
    """지금 data/로 manifest(data/versions/<버전>.json)와 dist/law-data-<버전>.zip을 만든다.

    Args:
        version: 버전 이름 (v1, v2, …).
        description: 이 버전 설명 (무엇이 바뀌었는지).
        force: 이미 있는 manifest를 덮어쓴다. Drive에 올린 버전에는 쓰지 않는다.
    """
    data_dir = get_settings().data_dir
    path = manifest_path(version, data_dir)
    if path.exists() and not force:
        raise SystemExit(f"이미 있는 버전입니다: {path}. 데이터가 바뀌었으면 새 버전을 만드세요.")
    if path.exists() and (dv.read_manifest(path).get("zip") or {}).get("drive_file_id"):
        raise SystemExit(f"{version}은 이미 Drive에 올렸습니다. 고치지 말고 새 버전을 만드세요.")
    try:
        manifest = dv.build_manifest(data_dir, version, description=description)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    for other in dv.load_manifests(dv.versions_dir(data_dir)):
        if other["version"] != version and not dv.local_diff(other, data_dir):
            raise SystemExit(f"지금 data/는 {other['version']}과 같습니다. 새 버전을 만들 필요가 없습니다.")

    output = PROJECT_ROOT / "dist" / dv.zip_name(version)
    write_zip(manifest, data_dir, output)
    manifest["zip"] |= {"sha256": dv.sha256sum(output), "bytes": output.stat().st_size}
    manifest["drive_folder"] = DEFAULT_FOLDER_URL
    dv.write_manifest(manifest, path)

    for role, entry in manifest["files"].items():
        rows = f"{entry['rows']:>6}행" if "rows" in entry else ""
        print(f"  {role:10s} {entry['path']:28s} {rows}  {entry['sha256'][:12]}…")
    print(f"완료: {path}, {output} ({output.stat().st_size / 1e6:.1f}MB)")


def _drive_file(remote: str, folder: str, name: str) -> dict | None:
    """Drive 폴더에서 이름이 같은 파일 정보(rclone lsjson 한 줄). 없으면 None."""
    out = subprocess.run(
        ["rclone", "lsjson", f"{remote}:", f"--drive-root-folder-id={folder}", "--files-only",
         "--include", name],
        check=True, capture_output=True, text=True,
    ).stdout
    return next((f for f in json.loads(out or "[]") if f["Name"] == name), None)


@app.command
def upload(version: str, *, folder_url: str = DEFAULT_FOLDER_URL, remote: str = "gdrive") -> None:
    """dist/law-data-<버전>.zip을 Drive 폴더에 올리고 파일 id를 manifest에 적는다.

    같은 이름의 zip이 Drive에 이미 있으면 올리지 않는다 (버전은 고치지 않는다).

    Args:
        version: 버전 이름.
        folder_url: 올릴 Drive 폴더 링크 ("링크가 있는 모든 사용자" 공유).
        remote: rclone config에서 만든 Google Drive remote 이름.
    """
    manifest, path = _load(version)
    zip_path = PROJECT_ROOT / "dist" / manifest["zip"]["name"]
    if not zip_path.exists() or dv.sha256sum(zip_path) != manifest["zip"]["sha256"]:
        raise SystemExit(f"{zip_path}가 없거나 manifest와 다릅니다. create {version}을 다시 실행하세요.")
    if shutil.which("rclone") is None:
        raise SystemExit("rclone이 없습니다. `brew install rclone` 후 `rclone config`로 로그인하세요.")

    folder = folder_id(folder_url)
    if existing := _drive_file(remote, folder, zip_path.name):
        raise SystemExit(f"Drive에 이미 {zip_path.name}이 있습니다 (id {existing['ID']}). 새 버전을 만드세요.")
    print(f"업로드 중: {zip_path.name} → {remote}: (폴더 {folder})")
    # 한 번에 보내는 업로드는 마지막 응답에서 멈추곤 해서, 1MB 청크로 이어 올리기(resumable)를 쓴다.
    subprocess.run(
        ["rclone", "copyto", str(zip_path), f"{remote}:{zip_path.name}",
         f"--drive-root-folder-id={folder}", "--drive-upload-cutoff=1M", "--drive-chunk-size=1M",
         "--stats=10s", "--stats-one-line", "-v"],
        check=True,
    )
    uploaded = _drive_file(remote, folder, zip_path.name)
    if not uploaded:
        raise SystemExit("업로드한 파일을 Drive에서 찾지 못했습니다.")
    manifest["zip"]["drive_file_id"] = uploaded["ID"]
    manifest["drive_folder"] = folder_url
    dv.write_manifest(manifest, path)
    print(f"완료: {dv.source_url(manifest)} → {path} (커밋하세요)")


@app.command
def register(version: str, *, force: bool = False) -> None:
    """MLflow law-data 실험에 data/<버전> run을 남긴다 (역할별 데이터셋 + manifest).

    Args:
        version: 버전 이름.
        force: 같은 버전 run이 이미 있어도 하나 더 남긴다.
    """
    if not tracking.enabled():
        raise SystemExit("MLFLOW_TRACKING_URI가 없습니다 (.env). 예: MLFLOW_TRACKING_URI=http://127.0.0.1:5050")
    manifest, path = _load(version)
    if not manifest["zip"].get("drive_file_id"):
        print(f"경고: {version}을 아직 Drive에 올리지 않았습니다. 데이터셋 주소가 Drive 폴더가 됩니다.")
    if (run_id := tracking.find_data_run(version)) and not force:
        print(f"이미 등록했습니다: run {run_id} (다시 남기려면 --force)")
        return
    run_id = tracking.register_dataset(manifest, path)
    print(f"완료: {tracking.DATA_EXPERIMENT} 실험 run data/{version} ({run_id})")


def install(extracted: Path, manifest: dict, data_dir: Path, *, force: bool = False) -> None:
    """푼 버전 파일을 data/에 놓는다. 다른 내용의 파일을 덮어써야 하면 force가 필요하다.

    Raises:
        SystemExit: 받은 파일이 manifest와 다르거나, force 없이 덮어써야 할 때.
    """
    if broken := dv.local_diff(manifest, extracted):
        raise SystemExit(f"받은 zip이 manifest와 다릅니다: {broken}")
    conflicts = [d for d in dv.local_diff(manifest, data_dir) if not d.startswith("없음")]
    if conflicts and not force:
        shown = "\n  ".join(conflicts[:10])
        raise SystemExit(f"data/에 {manifest['version']}과 다른 파일이 있습니다 (덮어쓰려면 --force):\n  {shown}")
    for entry in manifest["files"].values():
        if entry["path"].endswith("/"):
            target = data_dir / entry["path"]
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(extracted / entry["path"], target)
        else:
            (data_dir / entry["path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extracted / entry["path"], data_dir / entry["path"])


@app.command
def pull(version: str, *, force: bool = False) -> None:
    """Drive에서 버전 zip을 받아 sha256을 확인하고 data/에 푼다.

    Args:
        version: 버전 이름 (data/versions/<버전>.json이 git에 있어야 한다).
        force: data/에 다른 내용의 파일이 있어도 덮어쓴다.
    """
    manifest, _ = _load(version)
    data_dir = get_settings().data_dir
    if not dv.local_diff(manifest, data_dir):
        print(f"이미 {version}과 같습니다: {data_dir}")
        return
    file_id = manifest["zip"].get("drive_file_id")
    if not file_id:
        raise SystemExit(f"{version}은 아직 Drive에 올리지 않았습니다 (upload {version}).")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / manifest["zip"]["name"]
        if not gdown.download(id=file_id, output=str(zip_path), quiet=False):
            raise SystemExit("다운로드 실패. 파일 공유가 '링크가 있는 모든 사용자'인지 확인하세요.")
        if (actual := dv.sha256sum(zip_path)) != manifest["zip"]["sha256"]:
            raise SystemExit(f"sha256 불일치 (기대: {manifest['zip']['sha256']}, 실제: {actual}).")
        extracted = extract_zip(zip_path, Path(tmp) / "extracted")
        install(extracted, manifest, data_dir, force=force)
    print(f"완료: {version} → {data_dir}")


@app.command
def status() -> None:
    """버전마다 지금 data/와 같은지, 다르면 무엇이 다른지 보여 준다."""
    data_dir = get_settings().data_dir
    manifests = dv.load_manifests(dv.versions_dir(data_dir))
    if not manifests:
        print(f"버전이 없습니다: {dv.versions_dir(data_dir)}")
        return
    for manifest in manifests:
        diffs = dv.local_diff(manifest, data_dir)
        uploaded = "Drive" if manifest["zip"].get("drive_file_id") else "로컬만"
        print(f"{manifest['version']} ({manifest['created']}, {uploaded}): {'일치' if not diffs else f'다름 {len(diffs)}건'}")
        for d in diffs[:10]:
            print(f"    {d}")


if __name__ == "__main__":
    app()
