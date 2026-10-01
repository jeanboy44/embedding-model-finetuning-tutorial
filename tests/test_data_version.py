"""데이터 버전(ragkit.training.data_version, scripts/data_version.py) 테스트. Drive 접속 없이 돈다."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from ragkit.config import get_settings
from ragkit.training import data_version as dv

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
spec = importlib.util.spec_from_file_location("data_version_script", SCRIPTS_DIR / "data_version.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """버전에 넣을 파일이 다 있는 data/."""
    data = tmp_path / "data"
    (data / "processed").mkdir(parents=True)
    (data / "processed" / "law_docs.json").write_text(json.dumps([{"id": "a"}, {"id": "b"}]), encoding="utf-8")
    _write_jsonl(data / "splits" / "train.jsonl", [{"query": "q1"}, {"query": "q2"}])
    _write_jsonl(data / "splits" / "dev.jsonl", [{"query": "q3"}])
    _write_jsonl(data / "splits" / "test.jsonl", [{"query": "q4"}])
    (data / "splits" / "split_meta.json").write_text("{}", encoding="utf-8")
    _write_jsonl(data / "questions" / "최저임금법__p01.jsonl", [{"query": "q1"}, {"query": "q4"}])
    _write_jsonl(data / "questions" / "도로교통법__p01.jsonl", [{"query": "q2"}, {"query": "q3"}])
    return data


def test_build_manifest_hashes_and_counts(data_dir) -> None:
    manifest = dv.build_manifest(data_dir, "v1", description="첫 버전")

    files = manifest["files"]
    assert manifest["name"] == "law-retrieval" and manifest["version"] == "v1"
    assert files["corpus"]["rows"] == 2
    assert [files[s]["rows"] for s in ("train", "dev", "test")] == [2, 1, 1]
    assert files["corpus"]["sha256"] == dv.sha256sum(data_dir / "processed" / "law_docs.json")
    assert files["questions"]["rows"] == 4
    assert sorted(files["questions"]["files"]) == ["도로교통법__p01.jsonl", "최저임금법__p01.jsonl"]
    assert "rows" not in files["split_meta"]


def test_build_manifest_requires_all_files(data_dir) -> None:
    (data_dir / "splits" / "dev.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="splits/dev.jsonl"):
        dv.build_manifest(data_dir, "v1")


def test_local_diff_reports_changes(data_dir) -> None:
    manifest = dv.build_manifest(data_dir, "v1")
    assert dv.local_diff(manifest, data_dir) == []

    _write_jsonl(data_dir / "splits" / "test.jsonl", [{"query": "바뀐 질문"}])
    _write_jsonl(data_dir / "questions" / "새법__p01.jsonl", [{"query": "q"}])
    (data_dir / "questions" / "도로교통법__p01.jsonl").unlink()

    assert dv.local_diff(manifest, data_dir) == [
        "다름: splits/test.jsonl",
        "없음: questions/도로교통법__p01.jsonl",
        "추가됨: questions/새법__p01.jsonl",
    ]


def test_match_files_finds_version_by_content(data_dir, tmp_path) -> None:
    """경로가 아니라 내용으로 찾는다: 복사한 test 파일도 v1/test, 바뀐 파일은 unversioned."""
    manifest = dv.build_manifest(data_dir, "v1")
    manifest["zip"]["drive_file_id"] = "FILE123"
    dv.write_manifest(manifest, dv.versions_dir(data_dir) / "v1.json")
    copied = tmp_path / "my_questions.jsonl"
    copied.write_bytes((data_dir / "splits" / "test.jsonl").read_bytes())
    other = tmp_path / "other.jsonl"
    _write_jsonl(other, [{"query": "버전에 없는 질문"}])

    found, unmatched = dv.match_files(
        {"corpus": data_dir / "processed" / "law_docs.json", "questions": copied, "extra": other},
        dv.versions_dir(data_dir),
    )

    assert [(r.name, r.version) for r in found] == [("law-retrieval/corpus", "v1"), ("law-retrieval/test", "v1")]
    assert found[0].source == "https://drive.google.com/uc?id=FILE123"
    assert len(found[0].digest) == dv.DIGEST_LEN
    assert unmatched == ["extra"]
    assert dv.version_tag(found, unmatched) == "v1+unversioned"
    assert dv.version_tag(found, []) == "v1"
    assert dv.version_tag([], ["corpus"]) == "unversioned"


def test_load_manifests_orders_by_number(tmp_path) -> None:
    for v in ("v10", "v2", "v1"):
        dv.write_manifest({"version": v}, tmp_path / f"{v}.json")
    assert [m["version"] for m in dv.load_manifests(tmp_path)] == ["v1", "v2", "v10"]


def test_zip_extracts_to_data_layout_and_installs(data_dir, tmp_path) -> None:
    """zip은 law-data-v1/<data 아래 경로>. 받은 뒤 빈 data/에 그대로 놓인다."""
    manifest = dv.build_manifest(data_dir, "v1")
    zip_path = tmp_path / "law-data-v1.zip"
    script.write_zip(manifest, data_dir, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        assert "law-data-v1/processed/law_docs.json" in zf.namelist()
        assert "law-data-v1/questions/최저임금법__p01.jsonl" in zf.namelist()

    extracted = script.extract_zip(zip_path, tmp_path / "extracted")
    fresh = tmp_path / "fresh"
    script.install(extracted, manifest, fresh)
    assert dv.local_diff(manifest, fresh) == []


def test_install_refuses_to_overwrite_without_force(data_dir, tmp_path) -> None:
    manifest = dv.build_manifest(data_dir, "v1")
    zip_path = tmp_path / "law-data-v1.zip"
    script.write_zip(manifest, data_dir, zip_path)
    extracted = script.extract_zip(zip_path, tmp_path / "extracted")
    _write_jsonl(data_dir / "splits" / "train.jsonl", [{"query": "로컬에서 고친 질문"}])
    _write_jsonl(data_dir / "questions" / "새법__p01.jsonl", [{"query": "q"}])

    with pytest.raises(SystemExit, match="--force"):
        script.install(extracted, manifest, data_dir)

    script.install(extracted, manifest, data_dir, force=True)
    assert dv.local_diff(manifest, data_dir) == []  # 버전에 없는 질문 파일도 지운다


def test_create_writes_manifest_and_refuses_duplicates(data_dir, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAGKIT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    get_settings.cache_clear()
    try:
        script.create("v1", description="첫 버전")
        manifest = dv.read_manifest(data_dir / "versions" / "v1.json")
        assert manifest["zip"]["sha256"] == dv.sha256sum(tmp_path / "dist" / "law-data-v1.zip")
        assert manifest["description"] == "첫 버전"

        with pytest.raises(SystemExit, match="이미 있는 버전"):
            script.create("v1")
        with pytest.raises(SystemExit, match="v1과 같습니다"):
            script.create("v2")
    finally:
        get_settings.cache_clear()
