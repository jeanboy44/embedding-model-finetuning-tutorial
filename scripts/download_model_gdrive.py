"""Google Drive에서 임베딩 모델 zip을 내려받아 로컬 models/ 폴더에 압축을 푼다.

HuggingFace Hub 접속이 어려운 환경(사내망, 프록시 등)을 위한 대안이다.

사용법:
    uv run python scripts/download_model_gdrive.py
    uv run python scripts/download_model_gdrive.py --force

저장된 모델은 ragkit.models.load_embedding_model이 자동으로 찾아 사용한다.
"""

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

import gdown
import requests
from cyclopts import run
from rich.console import Console
from rich.markup import escape
from rich.progress import DownloadColumn, Progress, TransferSpeedColumn

from ragkit.config import get_settings

# 강사가 Drive에 올린 dist/multilingual-e5-small.zip ("링크가 있는 모든 사용자" 공유)
DEFAULT_FILE_ID = "1q0q5-5g_hpVqUrTzy3-ccX22UjqI5K6c"
DEFAULT_SHA256 = "8930c2ec9527c734d5afa10c77ef31a2ed5aac74c16387613a952994cb9d7281"
console = Console(emoji=False, soft_wrap=True)  # 긴 경로·URL을 중간에서 끊지 않게


def sha256sum(path: Path) -> str:
    """파일의 sha256 해시를 계산한다.

    Args:
        path: 대상 파일 경로.

    Returns:
        16진수 sha256 문자열.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    """zip을 풀어 모델 파일을 output_dir에 둔다.

    zip 안에 최상위 폴더가 하나만 있으면(예: multilingual-e5-small/) 그 내용을 옮긴다.
    zip 밖 경로를 가리키는 항목(../ 등)이 있으면 거부한다.

    Args:
        zip_path: 압축 파일 경로.
        output_dir: 모델 파일이 놓일 디렉토리. 이미 있으면 교체한다.

    Returns:
        output_dir.

    Raises:
        ValueError: zip에 안전하지 않은 경로가 있을 때.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as tmp:
        tmp_dir = Path(tmp).resolve()
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if not (tmp_dir / name).resolve().is_relative_to(tmp_dir):
                    raise ValueError(f"안전하지 않은 zip 경로: {name}")
            zf.extractall(tmp_dir)

        entries = list(tmp_dir.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp_dir

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(root), str(output_dir))
    return output_dir


# 큰 파일은 drive.google.com/uc 가 "바이러스 검사 경고" 페이지를 주고, gdown은 그 페이지의
# 확인 링크(drive.google.com/uc?...&confirm=t&uuid=...)를 따라간다. 이 링크가 Google의
# "자동화된 요청" 차단 페이지를 받는 일이 있어, 확인 없이 바로 주는 주소로 먼저 받는다.
DIRECT_URL = "https://drive.usercontent.google.com/download"


def drive_download(file_id: str, output: Path) -> bool:
    """공개 Drive 파일을 output에 받는다. 직접 주소가 안 되면 gdown으로 다시 시도한다.

    Args:
        file_id: Drive 파일 ID.
        output: 저장할 파일 경로.

    Returns:
        받았으면 True.
    """
    params = {"id": file_id, "export": "download", "confirm": "t"}
    columns = (*Progress.get_default_columns(), DownloadColumn(), TransferSpeedColumn())
    try:
        with requests.get(DIRECT_URL, params=params, stream=True, timeout=60) as res:
            res.raise_for_status()
            if "attachment" not in res.headers.get("Content-Disposition", ""):
                raise RuntimeError("파일 대신 HTML 페이지를 받았습니다")
            total = int(res.headers.get("Content-Length", 0)) or None
            with output.open("wb") as f, Progress(*columns, console=console) as progress:
                task = progress.add_task(output.name, total=total)
                for chunk in res.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))
        return True
    except (requests.RequestException, RuntimeError) as e:
        console.print(f"[yellow]직접 받기 실패 ({escape(str(e))}). gdown으로 다시 시도합니다.[/]")
        output.unlink(missing_ok=True)
    return gdown.download(id=file_id, output=str(output)) is not None


def download_from_gdrive(
    file_id: str,
    output_dir: Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Google Drive에서 모델 zip을 받아 검증 후 압축을 푼다.

    Args:
        file_id: Drive 파일 ID 또는 공유 링크.
        output_dir: 모델을 저장할 디렉토리.
        expected_sha256: 기대하는 zip의 sha256. None이면 검증을 건너뛴다.
        force: True면 이미 받은 모델이 있어도 다시 받는다.

    Returns:
        모델이 저장된 디렉토리 경로.

    Raises:
        ValueError: 파일 ID가 비어 있거나 sha256이 일치하지 않을 때.
        RuntimeError: 다운로드에 실패했을 때.
    """
    if (output_dir / "config.json").exists() and not force:
        console.print(f"이미 존재합니다: [dim]{escape(str(output_dir))}[/] (다시 받으려면 --force)")
        return output_dir

    if not file_id:
        raise ValueError(
            "Drive 파일 ID가 없습니다. --file-id로 지정하거나 DEFAULT_FILE_ID를 채우세요."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    try:
        console.print(f"[dim]다운로드 중: Google Drive → {escape(str(zip_path))}[/]")
        if file_id.startswith("http"):
            result = gdown.download(url=file_id, output=str(zip_path))
        else:
            result = str(zip_path) if drive_download(file_id, zip_path) else None
        if result is None or not zip_path.exists():
            raise RuntimeError(
                "다운로드 실패. 공유 설정이 '링크가 있는 모든 사용자'인지 확인하세요."
            )

        if expected_sha256:
            actual = sha256sum(zip_path)
            if actual != expected_sha256:
                raise ValueError(
                    f"sha256 불일치 (기대: {expected_sha256}, 실제: {actual}). "
                    "파일이 손상되었거나 다른 파일입니다."
                )
            console.print("[green]sha256 확인 완료[/]")

        console.print(f"[dim]압축 해제 중: {escape(str(output_dir))}[/]")
        extract_zip(zip_path, output_dir)
    finally:
        zip_path.unlink(missing_ok=True)

    console.print(f"완료: [dim]{escape(str(output_dir))}[/]")
    return output_dir


def main(
    file_id: str = DEFAULT_FILE_ID,
    output_dir: Path | None = None,
    sha256: str = DEFAULT_SHA256,
    force: bool = False,
) -> None:
    """Google Drive에서 임베딩 모델을 내려받는다.

    Args:
        file_id: Drive 파일 ID 또는 공유 링크.
        output_dir: 저장 디렉토리. 기본값은 models/<모델 이름>.
        sha256: zip 파일의 기대 sha256. 빈 문자열이면 검증하지 않는다.
        force: 이미 받은 모델이 있어도 다시 받는다.
    """
    settings = get_settings()
    output_dir = output_dir or (
        settings.models_dir / settings.embedding_model_name.split("/")[-1]
    )
    download_from_gdrive(
        file_id, output_dir, expected_sha256=sha256 or None, force=force
    )


if __name__ == "__main__":
    run(main)
