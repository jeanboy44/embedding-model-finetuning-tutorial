"""파인튜닝 모델 Drive 묶기 · 풀기 테스트 (네트워크 없음)."""

import importlib.util
import os
import zipfile
from pathlib import Path

import pytest

from ragkit.retrieval import model_key

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_model(root: Path, name: str, *, with_index: bool = True) -> tuple[Path, Path]:
    model_dir = root / "models" / "finetuned" / name
    (model_dir / "1_Pooling").mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "config.json").write_text("{}")
    (model_dir / "1_Pooling" / "config.json").write_text("{}")
    os.utime(model_dir / "model.safetensors", (1_759_000_000, 1_759_000_000))
    index = (
        root / "data" / "processed" / "index" / f"{model_key(name, model_dir)}.sqlite"
    )
    if with_index:
        index.parent.mkdir(parents=True)
        index.write_bytes(b"index")
    return model_dir, index


def test_bundle_keeps_repo_relative_paths(tmp_path) -> None:
    """zip 안 경로가 저장소 루트 기준이고, 인덱스가 함께 들어간다."""
    drive = _load_script("finetuned_drive")
    _, index = _fake_model(tmp_path, "r001_A")

    entry = drive.bundle("r001_A", tmp_path, tmp_path / "dist" / "finetuned-r001_A.zip")

    with zipfile.ZipFile(tmp_path / "dist" / "finetuned-r001_A.zip") as zf:
        names = set(zf.namelist())
    assert "models/finetuned/r001_A/model.safetensors" in names
    assert "models/finetuned/r001_A/1_Pooling/config.json" in names
    assert index.relative_to(tmp_path).as_posix() in names
    assert entry["index"] == index.relative_to(tmp_path).as_posix()
    assert entry["model_key"] == index.stem


def test_bundle_without_index(tmp_path) -> None:
    """인덱스가 없으면 모델만 묶는다."""
    drive = _load_script("finetuned_drive")
    _fake_model(tmp_path, "exp_002", with_index=False)

    entry = drive.bundle("exp_002", tmp_path, tmp_path / "out.zip")

    assert entry["index"] is None


def test_install_restores_model_key(tmp_path) -> None:
    """푼 뒤 가중치 수정 시각을 되돌려, 모델 키가 받은 인덱스 이름과 같다."""
    drive = _load_script("finetuned_drive")
    src, dst = tmp_path / "src", tmp_path / "dst"
    _fake_model(src, "r001_A")
    zip_path = tmp_path / "m.zip"
    entry = drive.bundle("r001_A", src, zip_path)

    drive.install(zip_path, entry, dst)

    model_dir = dst / "models" / "finetuned" / "r001_A"
    assert (model_dir / "model.safetensors").read_bytes() == b"weights"
    assert model_key("r001_A", model_dir) == entry["model_key"]
    assert (dst / entry["index"]).exists()


def test_install_rejects_unsafe_paths(tmp_path) -> None:
    """zip 밖을 가리키는 경로는 거부한다."""
    drive = _load_script("finetuned_drive")
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../evil.txt", "x")
    entry = {"model_dir": "models/finetuned/x", "weights_mtime": 0}

    with pytest.raises(ValueError):
        drive.install(zip_path, entry, tmp_path / "root")


def test_download_skips_existing(tmp_path, monkeypatch) -> None:
    """모델과 인덱스가 이미 있으면 받지 않는다."""
    drive = _load_script("finetuned_drive")
    _, index = _fake_model(tmp_path, "r001_A")
    manifest = {
        "drive_folder": drive.DEFAULT_FOLDER_URL,
        "models": {
            "r001_A": {
                "file": "finetuned-r001_A.zip",
                "file_id": "abc",
                "model_dir": "models/finetuned/r001_A",
                "index": index.relative_to(tmp_path).as_posix(),
                "model_key": index.stem,
                "weights_mtime": 1_759_000_000,
                "sha256": "",
                "size_mb": 1.0,
            }
        },
        "indexes": {},
    }
    monkeypatch.setattr(drive, "load_manifest", lambda: manifest)
    monkeypatch.setenv("RAGKIT_PROJECT_ROOT", str(tmp_path))
    from ragkit.config import get_settings

    get_settings.cache_clear()
    calls: list = []
    monkeypatch.setattr(drive.gdown, "download", lambda **kw: calls.append(kw))
    try:
        drive.download()
    finally:
        get_settings.cache_clear()

    assert calls == []


def test_index_only_bundle_roundtrip(tmp_path) -> None:
    """인덱스만 묶고 풀 수 있다 (학습 전 e5 인덱스)."""
    drive = _load_script("finetuned_drive")
    index = (
        tmp_path
        / "src"
        / "data"
        / "processed"
        / "index"
        / "multilingual-e5-small.sqlite"
    )
    index.parent.mkdir(parents=True)
    index.write_bytes(b"base-index")
    zip_path = tmp_path / "index.zip"

    entry = drive.bundle_index("multilingual-e5-small", tmp_path / "src", zip_path)
    drive.install(zip_path, entry, tmp_path / "dst")

    assert (tmp_path / "dst" / entry["index"]).read_bytes() == b"base-index"
