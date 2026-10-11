"""실험 한 바퀴의 기록(runs/<실험>/, experiments/<실험>/)을 MLflow에 올린다.

ragkit train · evaluate는 MLFLOW_TRACKING_URI가 있으면 스스로 기록한다. 이 스크립트는 그 위에
바퀴 단위 기록을 더한다: 바퀴 run 하나(부모)와 학습 run마다 자식 run 하나.

  부모 run "loop/<실험>"   tag: 가설 · 결정 · git commit · 상태 / artifact: 문서 여섯 개 · reviews · summary · timeline · manifest · digest
  자식 run "<실험>/<run>"  param: config.yaml(평탄화) / metric: dev · dev-b R@k, 대조군 대비 Δ · CI · p, 학습 시간 · best epoch
                           artifact: config · command · train_meta · eval JSON · paired_test · 로그

같은 실험을 다시 올리면 tag(loop_key)로 기존 run을 찾아 이어서 기록한다(중복 run을 만들지 않는다).

사용:
  uv run --extra mlflow python .claude/skills/experiment-loop/scripts/log_experiment.py <번호>_<이름>
  (MLFLOW_TRACKING_URI가 없으면 sqlite:///mlruns/mlflow.db를 쓴다)
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import yaml

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_URI = f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"
DOC_FILES = ["00_error_analysis.md", "01_research.md", "02_plan.md", "04_results.md", "05_retro.md"]
RUN_FILES = ["config.yaml", "command.sh", "train_meta.json", "eval_dev.json", "eval_dev_b.json",
             "paired_test.json", "paired_test_dev_b.json", "train.log", "eval_deploy.json"]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def eval_metrics(result: dict | None, prefix: str) -> dict:
    """ragkit evaluate JSON → {prefix.doc.recall_at_5: ...}"""
    if not result:
        return {}
    out = {}
    for judge in ("doc", "article", "multi"):
        for name, value in (result.get(judge) or {}).items():
            out[f"{prefix}.{judge}.{name.replace('@', '_at_')}"] = value
    for qtype, block in (result.get("by_query_type") or {}).items():
        if "doc" in block and "recall@5" in block["doc"]:
            out[f"{prefix}.type.{qtype}.recall_at_5"] = block["doc"]["recall@5"]
    return out


def paired_metrics(result: dict | None, prefix: str) -> dict:
    if not result:
        return {}
    out = {f"{prefix}.delta": result["delta"], f"{prefix}.ci_low": result["ci95"][0],
           f"{prefix}.ci_high": result["ci95"][1], f"{prefix}.fixed": result["fixed"],
           f"{prefix}.broken": result["broken"], f"{prefix}.mcnemar_p": result["mcnemar_p"]}
    by_law = result.get("by_law") or {}
    out[f"{prefix}.laws_worse"] = sum(1 for v in by_law.values() if v.get("delta", 0) < 0)
    return out


def find_or_start(name: str, key: str, parent_id: str | None = None) -> Any:
    query = f"tags.loop_key = '{key}'"
    found = mlflow.search_runs(filter_string=query, max_results=1, output_format="list")
    if found:
        return mlflow.start_run(run_id=found[0].info.run_id, nested=parent_id is not None)
    tags = {"loop_key": key}
    if parent_id:
        tags["mlflow.parentRunId"] = parent_id
    return mlflow.start_run(run_name=name, tags=tags, nested=parent_id is not None)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment", help="<번호>_<이름> (runs/와 experiments/ 아래 폴더 이름)")
    args = ap.parse_args()

    runs_dir = ROOT / "runs" / args.experiment
    docs_dir = ROOT / "experiments" / args.experiment
    if not runs_dir.exists():
        raise SystemExit(f"{runs_dir}가 없습니다")
    manifest = read_json(runs_dir / "manifest.json") or {}

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_URI)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT") or f"loop/{args.experiment}")

    with find_or_start(f"loop/{args.experiment}", f"loop/{args.experiment}") as parent:
        mlflow.set_tags({
            "hypothesis": manifest.get("hypothesis", ""),
            "decision": manifest.get("decision", ""),
            "status": manifest.get("status", ""),
            "split": manifest.get("split", ""),
            "git_commit": manifest.get("git_commit") or git_commit(),
        })
        for name in ["manifest.json", "summary.tsv", "queue.tsv", "timeline.md", "data_digest.json", "decision.json"]:
            if (runs_dir / name).exists():
                mlflow.log_artifact(str(runs_dir / name), "runs")
        for name in DOC_FILES:
            if (docs_dir / name).exists():
                mlflow.log_artifact(str(docs_dir / name), "docs")
        if (docs_dir / "reviews").exists():
            mlflow.log_artifacts(str(docs_dir / "reviews"), "docs/reviews")

        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists()):
            with find_or_start(f"{args.experiment}/{run_dir.name}", f"{args.experiment}/{run_dir.name}",
                               parent.info.run_id):
                config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
                params = {k: v for k, v in flatten(config).items() if v is not None}
                existing = mlflow.get_run(mlflow.active_run().info.run_id).data.params
                mlflow.log_params({k: v for k, v in params.items() if k not in existing})
                meta = read_json(run_dir / "train_meta.json") or {}
                metrics = {
                    **eval_metrics(read_json(run_dir / "eval_dev.json"), "dev"),
                    **eval_metrics(read_json(run_dir / "eval_dev_b.json"), "dev_b"),
                    **eval_metrics(read_json(run_dir / "eval_deploy.json"), "deploy"),
                    **paired_metrics(read_json(run_dir / "paired_test.json"), "vs_control"),
                    **paired_metrics(read_json(run_dir / "paired_test_dev_b.json"), "vs_control_dev_b"),
                }
                if meta:
                    metrics["train.seconds"] = meta.get("seconds", 0)
                    metrics["train.best_epoch"] = meta.get("best_epoch", 0)
                    metrics["train.examples"] = meta.get("train_examples", 0)
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
                for name in RUN_FILES:
                    if (run_dir / name).exists():
                        mlflow.log_artifact(str(run_dir / name))
        print(f"MLflow: experiment loop/{args.experiment}, parent run {parent.info.run_id}")


if __name__ == "__main__":
    main()
