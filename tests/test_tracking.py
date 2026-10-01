"""MLflow 연동(ragkit.tracking): 꺼져 있으면 아무 일도 안 하고, 켜면 실험·모델을 기록한다."""

from pathlib import Path

import pytest

from ragkit import tracking
from ragkit.config import get_settings

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def mlflow_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """임시 SQLite 저장소로 MLflow를 켠다 (아티팩트는 cwd/mlruns에 쌓이므로 tmp로 옮긴다)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "test-exp")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def mlflow_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_disabled_is_noop(mlflow_off) -> None:
    """추적 주소가 없으면 기록 없이 그대로 동작한다 (서버 없는 수강생·테스트)."""
    assert not tracking.enabled()
    with tracking.run("r", params={"a": 1}) as run:
        run.log_metrics({"recall@5": 0.5})
        run.log_dict({"x": 1}, "x.json")
    span = tracking.start_span("s", inputs={"q": 1})
    span.end(outputs={"a": 1})
    assert run.id is None


def test_run_logs_params_metrics_and_artifacts(mlflow_on) -> None:
    """run 하나에 파라미터·지표·결과 파일이 남는다. 지표 이름의 @는 MLflow 규칙에 맞게 바꾼다."""
    with tracking.run("exp_005/e5", params={"model": "e5", "k": 5}, tags={"stage": "1"}) as run:
        run.log_metrics({"recall@5": 0.513, "mrr@10": 0.38})
        run.log_dict({"per_question": []}, "result.json")

    got = mlflow.get_run(run.id)
    assert got.data.params == {"model": "e5", "k": "5"}
    assert got.data.metrics == {"recall_at_5": 0.513, "mrr_at_10": 0.38}
    assert got.data.tags["stage"] == "1"
    assert got.info.run_name == "exp_005/e5"
    assert mlflow.get_experiment(got.info.experiment_id).name == "test-exp"


def test_nested_runs(mlflow_on) -> None:
    """비교 실험은 부모 run 아래 모델별 자식 run으로 묶는다."""
    with tracking.run("compare") as parent, tracking.run("e5", nested=True) as child:
        child.log_metrics({"recall@5": 0.5})
    assert mlflow.get_run(child.id).data.tags["mlflow.parentRunId"] == parent.id


def test_register_and_resolve_model(mlflow_on) -> None:
    """모델 폴더를 등록하고 별칭으로 되찾는다. 인덱스 키도 함께 돌려준다."""
    model_dir = mlflow_on / "e5-int8"
    (model_dir / "onnx").mkdir(parents=True)
    (model_dir / "onnx" / "model.onnx").write_bytes(b"onnx")
    (model_dir / "tokenizer.json").write_text("{}")

    version = tracking.register_model(
        model_dir, name="law-embedder", alias="champion", index_key="e5-int8", metrics={"recall@5": 0.515}
    )
    local, index_key = tracking.resolve_model("models:/law-embedder@champion", cache_dir=mlflow_on / "registry")

    assert version == "1"
    assert index_key == "e5-int8"
    assert (Path(local) / "onnx" / "model.onnx").read_bytes() == b"onnx"
    assert (Path(local) / "tokenizer.json").exists()


def test_resolve_ignores_cache_from_another_registry(mlflow_on) -> None:
    """같은 이름·버전이라도 다른 등록(다른 run)에서 받은 캐시면 다시 내려받는다."""
    model_dir = mlflow_on / "e5-int8"
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text('{"real": true}')
    stale = mlflow_on / "registry" / "law-embedder-v1"
    stale.mkdir(parents=True)
    (stale / "tokenizer.json").write_text("{}")

    tracking.register_model(model_dir, name="law-embedder", alias="champion")
    local, _ = tracking.resolve_model("models:/law-embedder@champion", cache_dir=mlflow_on / "registry")

    assert (Path(local) / "tokenizer.json").read_text() == '{"real": true}'
