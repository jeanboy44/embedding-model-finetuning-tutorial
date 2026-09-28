"""설정 경로 테스트."""

from pathlib import Path

import pytest

from ragkit.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_project_root_defaults_to_current_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """설치된 패키지 위치와 무관하게, 기본 루트는 명령을 실행한 폴더다."""
    monkeypatch.delenv("RAGKIT_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(REPO_ROOT)

    settings = Settings()

    assert settings.project_root == REPO_ROOT
    assert settings.data_dir == REPO_ROOT / "data"
    assert settings.models_dir == REPO_ROOT / "models"


def test_project_root_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """RAGKIT_PROJECT_ROOT 환경 변수로 루트를 바꿀 수 있다."""
    monkeypatch.setenv("RAGKIT_PROJECT_ROOT", str(tmp_path))

    settings = Settings()

    assert settings.project_root == tmp_path
    assert settings.models_dir == tmp_path / "models"
