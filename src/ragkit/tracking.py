"""MLflow 연동: 실험 추적 · 모델 레지스트리 · 서비스 트레이싱.

`MLFLOW_TRACKING_URI`(.env)가 비어 있거나 mlflow가 설치되지 않았으면 모든 함수가 아무 일도 하지 않는다.
그래서 서버 없이 실습하는 수강생·테스트·배포 코드가 MLflow 유무를 신경 쓰지 않아도 된다.

    with tracking.run("exp_005/e5", params={...}) as run:   # 실험 추적 (extra [mlflow])
        run.log_metrics({"recall@5": 0.51})
    span = tracking.start_span("answer", "CHAIN", inputs=..., session_id=nb_id)   # 트레이싱 ([tracing])
    span.end(outputs=...)
    tracking.register_model("models/e5-pruned-int8", name="law-embedder", alias="champion")  # 레지스트리 ([mlflow])
    run.log_data({"corpus": corpus_path, "train": train_path})   # 입력 데이터 버전 (data/versions/v1.json)

트레이스는 스트리밍 답변처럼 여러 스레드를 오가는 흐름도 끊기지 않도록
컨텍스트에 기대지 않는 스팬(start_span_no_context)으로 부모-자식을 직접 잇는다.
"""

import importlib.util
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ragkit.config import get_settings

_configured: tuple[str, str] | None = None


def enabled() -> bool:
    """추적 주소가 있고 mlflow(또는 mlflow-tracing)가 설치돼 있으면 True."""
    return bool(get_settings().mlflow_tracking_uri) and importlib.util.find_spec("mlflow") is not None


def _mlflow(experiment: str | None = None):
    """mlflow 모듈을 설정(주소·실험)한 뒤 돌려준다. 같은 설정이면 다시 하지 않는다."""
    global _configured
    import mlflow

    settings = get_settings()
    wanted = (settings.mlflow_tracking_uri, experiment or settings.mlflow_experiment)
    if _configured != wanted:
        mlflow.set_tracking_uri(wanted[0])
        mlflow.set_experiment(wanted[1])
        _configured = wanted
    return mlflow


def metric_name(name: str) -> str:
    """MLflow 지표 이름 규칙에 맞춘다: recall@5 → recall_at_5."""
    return re.sub(r"[^\w\-. /:]", "_", name.replace("@", "_at_"))


# ============================================================
# 실험 추적
# ============================================================
class Run:
    """열린 run. MLflow가 꺼져 있으면 모든 메서드가 아무 일도 하지 않고 id는 None."""

    def __init__(self, mlflow_module=None, run_id: str | None = None) -> None:
        self._mlflow = mlflow_module
        self.id = run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow and params:
            self._mlflow.log_params({k: v for k, v in params.items() if v is not None})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow and metrics:
            clean = {metric_name(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            self._mlflow.log_metrics(clean, step=step)

    def log_dict(self, data: Any, artifact_file: str) -> None:
        if self._mlflow:
            self._mlflow.log_dict(data, artifact_file)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        if self._mlflow and Path(path).exists():
            self._mlflow.log_artifact(str(path), artifact_path)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self._mlflow and tags:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_data(self, paths: dict[str, Path]) -> None:
        """입력 파일이 어느 데이터 버전인지 찾아 run 입력(Datasets)과 data_version 태그로 남긴다.

        Args:
            paths: 입력 이름(context) → 파일. 예: {"corpus": law_docs.json, "train": train.jsonl}.
        """
        if not self._mlflow:
            return
        from ragkit.config import get_settings
        from ragkit.training import data_version

        found, unmatched = data_version.match_files(paths, data_version.versions_dir(get_settings().data_dir))
        contexts = [c for c in paths if c not in unmatched]
        _log_inputs(self._mlflow, found, contexts)
        self.set_tags({"data_version": data_version.version_tag(found, unmatched)})


@contextmanager
def run(
    name: str,
    *,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    nested: bool = False,
    experiment: str | None = None,
) -> Iterator[Run]:
    """실험 run 하나를 연다 (꺼져 있으면 아무 일도 하지 않는 Run)."""
    if not enabled():
        yield Run()
        return
    mlflow = _mlflow(experiment)
    with mlflow.start_run(run_name=name, nested=nested) as active:
        current = Run(mlflow, active.info.run_id)
        current.log_params(params or {})
        current.set_tags(tags or {})
        yield current


def flat_metrics(result: dict, prefix: str = "") -> dict[str, float]:
    """평가 결과(evaluate_retrieval 형식)를 MLflow 지표 딕셔너리로 편다.

    doc 지표는 이름 그대로(recall@5), article은 article/ 접두어, 질문 유형·테마별 R@5는 by_query_type/…
    """
    out = {f"{prefix}{k}": v for k, v in result.get("doc", {}).items()}
    out |= {f"{prefix}article/{k}": v for k, v in result.get("article", {}).items()}
    for group in ("by_query_type", "by_theme"):
        for name, sub in result.get(group, {}).items():
            if "recall@5" in sub.get("doc", {}):
                out[f"{prefix}{group}/{name}/recall@5"] = sub["doc"]["recall@5"]
    return out


# ============================================================
# 데이터 버전 (extra [mlflow])
# ============================================================
DATA_EXPERIMENT = "law-data"


def _log_inputs(mlflow, refs: list, contexts: list[str]) -> None:
    """DatasetRef를 MLflow 데이터셋(메타데이터만: 이름·digest·Drive 주소)으로 run 입력에 단다."""
    if not refs:
        return
    from mlflow.data.http_dataset_source import HTTPDatasetSource
    from mlflow.data.meta_dataset import MetaDataset

    mlflow.log_inputs(
        datasets=[MetaDataset(HTTPDatasetSource(r.source), name=r.name, digest=r.digest) for r in refs],
        contexts=contexts,
        tags_list=[{"data_version": r.version} for r in refs],
    )


def find_data_run(version: str) -> str | None:
    """law-data 실험에서 이 버전을 등록한 run id (없으면 None)."""
    mlflow = _mlflow(DATA_EXPERIMENT)
    found = mlflow.search_runs(
        experiment_names=[DATA_EXPERIMENT], filter_string=f"tags.data_version = '{version}'",
        output_format="list",
    )
    return found[0].info.run_id if found else None


def register_dataset(manifest: dict, manifest_path: Path) -> str | None:
    """데이터 버전 하나를 law-data 실험의 run(data/<버전>)으로 남긴다.

    역할(corpus·train·dev·test·questions)마다 데이터셋을 run 입력으로 달고,
    행 수는 파라미터, manifest는 아티팩트로 남긴다. 파일 자체는 올리지 않는다(Drive zip).

    Returns:
        run id. MLflow가 꺼져 있으면 None.
    """
    if not enabled():
        return None
    from ragkit.training import data_version

    mlflow = _mlflow(DATA_EXPERIMENT)
    version = manifest["version"]
    roles = [role for role in manifest["files"] if role != "split_meta"]
    with mlflow.start_run(run_name=f"data/{version}", description=manifest.get("description") or None) as active:
        current = Run(mlflow, active.info.run_id)
        _log_inputs(mlflow, [data_version.to_ref(manifest, r) for r in roles], roles)
        current.log_params({f"{r}.rows": manifest["files"][r].get("rows") for r in roles})
        current.set_tags({
            "data_version": version,
            "dataset": manifest["name"],
            "source": data_version.source_url(manifest),
            "zip_sha256": (manifest.get("zip") or {}).get("sha256") or "",
        })
        current.log_artifact(manifest_path)
    return active.info.run_id


# ============================================================
# 모델 레지스트리 (extra [mlflow])
# ============================================================
def register_model(
    model_dir: Path | str,
    *,
    name: str,
    alias: str | None = None,
    index_key: str | None = None,
    metrics: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
) -> str | None:
    """모델 폴더(토크나이저 + 가중치/ONNX)를 레지스트리에 새 버전으로 등록한다.

    Args:
        model_dir: 등록할 모델 폴더.
        name: 등록 모델 이름 (예: law-embedder).
        alias: 붙일 별칭 (예: champion). 앱은 models:/<name>@<alias>로 불러온다.
        index_key: 이 모델로 만든 인덱스 파일의 키 (data/processed/index/<키>.sqlite). 버전 태그로 남긴다.
        metrics: 함께 남길 평가 지표 (run 지표 + 버전 태그).
        params: 함께 남길 파라미터.

    Returns:
        등록한 버전 번호. MLflow가 꺼져 있으면 None.
    """
    if not enabled():
        return None
    mlflow = _mlflow()
    from mlflow import pyfunc

    model_dir = Path(model_dir)
    with mlflow.start_run(run_name=f"register/{name}") as active:
        current = Run(mlflow, active.info.run_id)
        current.log_params({"model_dir": str(model_dir), **(params or {})})
        current.log_metrics(metrics or {})
        info = pyfunc.log_model(
            name="model",
            python_model=_ModelFolder(),
            artifacts={"model_dir": str(model_dir)},
            registered_model_name=name,
        )
    version = str(info.registered_model_version)
    client = mlflow.MlflowClient()
    tags = {"index_key": index_key or model_dir.name, "source_dir": str(model_dir)}
    tags |= {metric_name(k): f"{v:.4f}" for k, v in (metrics or {}).items() if isinstance(v, (int, float))}
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    if alias:
        client.set_registered_model_alias(name, alias, version)
    return version


def resolve_model(uri: str, cache_dir: Path | None = None) -> tuple[str, str]:
    """models:/<이름>@<별칭> 또는 models:/<이름>/<버전>을 로컬 모델 폴더와 인덱스 키로 바꾼다.

    내려받은 폴더는 models/registry/<이름>-v<버전>/에 두고 다음부터 재사용한다.

    Raises:
        RuntimeError: MLflow가 꺼져 있을 때.
    """
    if not enabled():
        raise RuntimeError(f"{uri}를 쓰려면 MLFLOW_TRACKING_URI가 필요합니다 (.env)")
    mlflow = _mlflow()
    client = mlflow.MlflowClient()
    body = uri.removeprefix("models:/")
    if "@" in body:
        name, alias = body.split("@", 1)
        mv = client.get_model_version_by_alias(name, alias)
    else:
        name, number = body.rsplit("/", 1)
        mv = client.get_model_version(name, number)
    target = (cache_dir or get_settings().models_dir / "registry") / f"{name}-v{mv.version}"
    marker = target / ".mlflow_run_id"  # 같은 이름·버전이라도 다른 저장소의 등록이면 다시 받는다
    if not (marker.exists() and marker.read_text(encoding="utf-8") == mv.run_id):
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(mlflow.artifacts.download_artifacts(f"models:/{name}/{mv.version}", dst_path=tmp))
            folder = next(p.parent for p in downloaded.rglob("tokenizer.json"))
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(folder, target)
        marker.write_text(mv.run_id, encoding="utf-8")
    return str(target), mv.tags.get("index_key") or f"{name}-v{mv.version}"


try:  # 레지스트리 저장용 빈 모델 (추론은 ragkit이 폴더를 직접 읽는다)
    import mlflow.pyfunc as _pyfunc

    class _ModelFolder(_pyfunc.PythonModel):
        def predict(self, context, model_input: list[str], params=None) -> list[list[float]]:
            raise NotImplementedError("ragkit.embeddings.create_embedding_fn(모델 폴더)로 쓴다")
except ImportError:  # mlflow-tracing만 있거나 mlflow가 없을 때
    _ModelFolder = None  # type: ignore[assignment,misc]


# ============================================================
# 서비스 트레이싱 (extra [tracing] 또는 [mlflow])
# ============================================================
class Span:
    """열린 스팬. MLflow가 꺼져 있으면 아무 일도 하지 않는다."""

    def __init__(self, live=None) -> None:
        self._live = live

    @property
    def live(self):
        return self._live

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        if self._live:
            self._live.set_attributes(attributes)

    def end(self, outputs: Any = None, attributes: dict[str, Any] | None = None, error: str | None = None) -> None:
        if self._live:
            if error:
                attributes = {**(attributes or {}), "error": error}
            self._live.end(outputs=outputs, attributes=attributes, status="ERROR" if error else "OK")


def start_span(
    name: str,
    span_type: str = "CHAIN",
    *,
    parent: Span | None = None,
    inputs: Any = None,
    attributes: dict[str, Any] | None = None,
    session_id: str | None = None,
    user: str | None = None,
    tags: dict[str, str] | None = None,
) -> Span:
    """스팬을 연다. parent가 없으면 새 트레이스의 루트가 되고, 세션·사용자·태그를 트레이스에 남긴다."""
    if not enabled():
        return Span()
    mlflow = _mlflow(get_settings().mlflow_trace_experiment)
    parent_live = parent.live if parent else None
    metadata = None
    if parent_live is None:
        metadata = {k: v for k, v in {"mlflow.trace.session": session_id, "mlflow.trace.user": user}.items() if v}
    live = mlflow.start_span_no_context(
        name,
        span_type=span_type,
        parent_span=parent_live,
        inputs=inputs,
        attributes=attributes,
        tags=tags if parent_live is None else None,
        metadata=metadata or None,
    )
    return Span(live)
