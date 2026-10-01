"""DS CLI의 MLflow 실험 추적: train · evaluate · compare · register가 run을 남긴다."""

from pathlib import Path

import pytest

from ragkit.cli import train_cli
from ragkit.config import get_settings
from tests.test_train_cli import _fake_create, _write_inputs

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """임시 SQLite 저장소로 MLflow를 켜고, 실험의 run 목록(시작 순)을 돌려주는 함수를 준다."""
    monkeypatch.chdir(tmp_path)  # 아티팩트 기본 위치(cwd/mlruns)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "cli-test")
    get_settings.cache_clear()

    def fetch():
        exp = mlflow.get_experiment_by_name("cli-test")
        found = mlflow.search_runs([exp.experiment_id], output_format="list", order_by=["start_time ASC"])
        return found

    yield fetch
    get_settings.cache_clear()


def test_evaluate_logs_run_with_metrics(tmp_path, corpus, questions, monkeypatch, runs) -> None:
    """evaluate 한 번 = run 하나: 모델·분할 파라미터, R@5 등 지표, 결과 JSON."""
    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)

    train_cli.evaluate("intfloat/multilingual-e5-small", splits=tmp_path / "splits", corpus=corpus_path,
                       index=tmp_path / "e5.sqlite")

    [run] = runs()
    assert run.data.params["model"] == "intfloat/multilingual-e5-small"
    assert run.data.params["split"] == "test"
    assert "recall_at_5" in run.data.metrics
    assert "article/recall_at_5" in run.data.metrics


def test_compare_logs_parent_and_nested_model_runs(tmp_path, corpus, questions, monkeypatch, runs) -> None:
    """compare = 부모 run + 모델마다 자식 run. 부모에는 비교표를 남긴다."""
    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    exp = tmp_path / "experiments" / "exp_005"
    exp.mkdir(parents=True)
    config = exp / "config.yaml"
    config.write_text(
        f"name: base_model_comparison\nquestions: {qpath}\n"
        "models:\n  - intfloat/multilingual-e5-small\n  - google/embeddinggemma-300m\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)

    train_cli.compare(config, corpus=corpus_path, index_dir=tmp_path / "index")

    parent, *children = runs()
    assert parent.info.run_name == "base_model_comparison"
    assert [c.data.tags["mlflow.parentRunId"] for c in children] == [parent.info.run_id] * 2
    assert [c.data.params["model"] for c in children] == [
        "intfloat/multilingual-e5-small", "google/embeddinggemma-300m",
    ]
    assert all("recall_at_5" in c.data.metrics for c in children)
    artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(parent.info.run_id)]
    assert "comparison.md" in artifacts


def test_train_logs_config_and_dev_history(tmp_path, corpus, questions, monkeypatch, runs) -> None:
    """train = run 하나: 학습 설정 파라미터, epoch별 dev 지표(step=epoch), 학습 시간."""
    import ragkit.training.train as train_module

    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"training:\n  output_dir: {tmp_path / 'out'}\n  lr: 3.0e-5\n", encoding="utf-8")

    def fake_train(config, train_q, dev_q, docs, **kwargs):
        return {
            "train_examples": 10, "seconds": 12.0, "trainable_params": 1, "total_params": 2,
            "dev_before": {"dev_recall@5": 0.4},
            "dev_history": [{"epoch": 1.0, "dev_recall@5": 0.5}, {"epoch": 2.0, "dev_recall@5": 0.6}],
            "dev_after": {"dev_recall@5": 0.6}, "best_epoch": 2,
        }

    monkeypatch.setattr(train_module, "train", fake_train)

    train_cli.train(config_path, splits=tmp_path / "splits", corpus=corpus_path)

    [run] = runs()
    assert run.data.params["lr"] == "3e-05"
    assert run.data.metrics["train_seconds"] == 12.0
    assert run.data.metrics["best_epoch"] == 2
    history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "dev_recall_at_5")
    assert [(m.step, m.value) for m in history] == [(0, 0.4), (1, 0.5), (2, 0.6)]


def test_register_command_sets_alias(tmp_path, monkeypatch, runs) -> None:
    """ragkit register 폴더 --alias champion → 새 버전 + 별칭, 인덱스 키 태그."""
    from ragkit.cli import cli_tool

    model_dir = tmp_path / "e5-int8"
    (model_dir / "onnx").mkdir(parents=True)
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "onnx" / "model.onnx").write_bytes(b"onnx")

    cli_tool.register_command(model_dir, name="law-embedder", alias="champion")

    mv = mlflow.MlflowClient().get_model_version_by_alias("law-embedder", "champion")
    assert str(mv.version) == "1"
    assert mv.tags["index_key"] == "e5-int8"


def _write_version(tmp_path: Path, files: dict[str, Path]) -> None:
    """data/versions/v1.json: 주어진 파일들을 v1의 역할로 적는다."""
    from ragkit.training import data_version as dv

    manifest = {
        "name": "law-retrieval", "version": "v1", "created": "2026-10-01",
        "files": {role: {"path": str(p), "sha256": dv.sha256sum(p), "rows": 1} for role, p in files.items()},
        "zip": {"name": "law-data-v1.zip", "sha256": "abc", "drive_file_id": "FILE123"},
    }
    dv.write_manifest(manifest, tmp_path / "data" / "versions" / "v1.json")


def test_evaluate_links_data_version(tmp_path, corpus, questions, monkeypatch, runs) -> None:
    """입력 파일이 v1과 같으면 run 입력에 law-retrieval/corpus·test, 태그 data_version=v1."""
    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    _write_version(tmp_path, {"corpus": corpus_path, "test": tmp_path / "splits" / "test.jsonl"})
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)

    train_cli.evaluate("intfloat/multilingual-e5-small", splits=tmp_path / "splits", corpus=corpus_path,
                       index=tmp_path / "e5.sqlite")

    [run] = runs()
    assert run.data.tags["data_version"] == "v1"
    inputs = {d.dataset.name: d for d in run.inputs.dataset_inputs}
    assert sorted(inputs) == ["law-retrieval/corpus", "law-retrieval/test"]
    assert "FILE123" in inputs["law-retrieval/test"].dataset.source
    assert {t.key: t.value for t in inputs["law-retrieval/test"].tags}["mlflow.data.context"] == "test"


def test_evaluate_without_version_tags_unversioned(tmp_path, corpus, questions, monkeypatch, runs) -> None:
    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)

    train_cli.evaluate("intfloat/multilingual-e5-small", splits=tmp_path / "splits", corpus=corpus_path,
                       index=tmp_path / "e5.sqlite")

    [run] = runs()
    assert run.data.tags["data_version"] == "unversioned"
    assert run.inputs.dataset_inputs == []


def test_register_dataset_logs_data_run(tmp_path, monkeypatch, runs) -> None:
    """데이터 버전 등록 = law-data 실험의 data/v1 run: 역할별 데이터셋, 행 수, manifest 아티팩트."""
    from ragkit import tracking
    from ragkit.training import data_version as dv

    manifest_path = tmp_path / "v1.json"
    files = {role: {"path": role, "sha256": f"{role:0<64}", "rows": 3}
             for role in ("corpus", "train", "dev", "test", "split_meta", "questions")}
    manifest = {"name": "law-retrieval", "version": "v1", "description": "첫 버전", "files": files,
                "zip": {"sha256": "z" * 64, "drive_file_id": "FILE123"}}
    dv.write_manifest(manifest, manifest_path)

    assert tracking.find_data_run("v1") is None
    run_id = tracking.register_dataset(manifest, manifest_path)

    assert tracking.find_data_run("v1") == run_id
    run = mlflow.get_run(run_id)
    assert run.info.run_name == "data/v1"
    assert run.data.tags["data_version"] == "v1"
    assert run.data.params["train.rows"] == "3"
    names = sorted(d.dataset.name for d in run.inputs.dataset_inputs)
    assert names == [f"law-retrieval/{r}" for r in ("corpus", "dev", "questions", "test", "train")]
    assert [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)] == ["v1.json"]
