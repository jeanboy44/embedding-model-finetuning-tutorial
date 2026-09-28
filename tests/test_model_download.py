"""모델 다운로드 스크립트 및 로컬 모델 경로 해석 테스트."""

import importlib.util
import shutil
import zipfile
from pathlib import Path

import pytest

from ragkit.config import get_settings
from ragkit.models import resolve_model_source

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_from_hf_calls_snapshot(tmp_path, monkeypatch) -> None:
    """필요한 파일 패턴으로 snapshot_download를 호출한다."""
    hf = _load_script("download_model_hf")
    calls: list[dict] = []
    monkeypatch.setattr(hf, "snapshot_download", lambda **kw: calls.append(kw))

    out = hf.download_from_hf("org/model", tmp_path / "model")

    assert out == tmp_path / "model"
    assert calls[0]["repo_id"] == "org/model"
    assert calls[0]["local_dir"] == tmp_path / "model"
    assert "model.safetensors" in calls[0]["allow_patterns"]


@pytest.mark.parametrize(("force", "expected_calls"), [(False, 0), (True, 1)])
def test_download_from_hf_skips_existing(
    tmp_path, monkeypatch, force: bool, expected_calls: int
) -> None:
    """이미 받은 모델은 --force가 없으면 다시 받지 않는다."""
    hf = _load_script("download_model_hf")
    calls: list[dict] = []
    monkeypatch.setattr(hf, "snapshot_download", lambda **kw: calls.append(kw))
    (tmp_path / "config.json").write_text("{}")

    hf.download_from_hf("org/model", tmp_path, force=force)

    assert len(calls) == expected_calls


def test_resolve_model_source(tmp_path, monkeypatch) -> None:
    """로컬 models/<이름>이 있으면 경로를, 없으면 원래 이름을 반환한다."""
    settings = get_settings()
    monkeypatch.setattr(type(settings), "models_dir", property(lambda _: tmp_path))

    assert resolve_model_source("org/model") == "org/model"

    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    assert resolve_model_source("org/model") == str(tmp_path / "model")


def _make_model_zip(path: Path, top: str = "my-model") -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{top}/config.json", "{}")
        zf.writestr(f"{top}/1_Pooling/config.json", "{}")
    return path


def _fake_gdown(src_zip: Path):
    def _download(*, output: str, **_kw) -> str:
        shutil.copy(src_zip, output)
        return output

    return _download


def test_download_from_gdrive_extracts(tmp_path, monkeypatch) -> None:
    """zip을 받아 sha256 확인 후 output_dir에 풀고, zip은 지운다."""
    gd = _load_script("download_model_gdrive")
    src_zip = _make_model_zip(tmp_path / "src.zip")
    monkeypatch.setattr(gd.gdown, "download", _fake_gdown(src_zip))
    out_dir = tmp_path / "models" / "e5"

    gd.download_from_gdrive("file-id", out_dir, expected_sha256=gd.sha256sum(src_zip))

    assert (out_dir / "config.json").exists()
    assert (out_dir / "1_Pooling" / "config.json").exists()
    assert list((tmp_path / "models").iterdir()) == [out_dir]


def test_download_from_gdrive_sha_mismatch(tmp_path, monkeypatch) -> None:
    """sha256이 다르면 실패하고 모델도 zip도 남기지 않는다."""
    gd = _load_script("download_model_gdrive")
    src_zip = _make_model_zip(tmp_path / "src.zip")
    monkeypatch.setattr(gd.gdown, "download", _fake_gdown(src_zip))
    out_dir = tmp_path / "models" / "e5"

    with pytest.raises(ValueError, match="sha256"):
        gd.download_from_gdrive("file-id", out_dir, expected_sha256="0" * 64)

    assert list((tmp_path / "models").iterdir()) == []


def test_download_from_gdrive_requires_file_id(tmp_path) -> None:
    """파일 ID가 비어 있으면 안내 메시지와 함께 실패한다."""
    gd = _load_script("download_model_gdrive")
    with pytest.raises(ValueError, match="파일 ID"):
        gd.download_from_gdrive("", tmp_path / "e5")


def test_extract_zip_rejects_path_traversal(tmp_path) -> None:
    """zip 밖을 가리키는 경로는 풀지 않는다."""
    gd = _load_script("download_model_gdrive")
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")

    with pytest.raises(ValueError, match="안전하지 않은"):
        gd.extract_zip(bad, tmp_path / "out" / "e5")
    assert not (tmp_path / "out" / "evil.txt").exists()
