"""ragkit-bench: 모델 변형별 측정을 각각 새 프로세스에서 돌려 비교표를 만든다.

변형마다 프로세스를 새로 띄우는 이유: 최대 메모리(ru_maxrss)를 그 변형만의 값으로 재기 위해서다.
"""

import json
import multiprocessing as mp
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cyclopts

from .measure import install_size, measure, model_size
from .report import to_markdown
from .variants import DEFAULT_VARIANTS, Variant

app = cyclopts.App(name="ragkit-bench", help="원본 / ONNX / INT8 모델의 속도·메모리·크기·정확도 비교")


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # macOS는 바이트, Linux는 KB


def prepare_index(variant: Variant, corpus_path: str) -> str:
    """변형의 인덱스를 만든다(있으면 그대로). 측정과 다른 프로세스에서 불러 메모리 측정에 섞이지 않게 한다."""
    from ragkit.data import load_corpus
    from ragkit.embeddings import create_embedding_fn, get_profile
    from ragkit.retrieval import build_index, default_index_path, model_key

    corpus = load_corpus(Path(corpus_path))
    key = model_key(variant.model)
    embed_fn = create_embedding_fn(variant.model, backend=variant.backend, device=variant.device)
    build_index(corpus, embed_fn, default_index_path(key), model_key=key,
                format_doc=get_profile(variant.model).format_doc)
    return str(default_index_path(key))


def run_variant(variant: Variant, corpus_path: str, questions_path: str, index_path: str, n_latency: int) -> dict:
    """한 변형을 로드하고 측정한다. 새 프로세스에서 불러 최대 메모리가 이 변형만의 값이 되게 한다."""
    from ragkit.data import filter_questions, load_corpus, load_questions
    from ragkit.embeddings import create_embedding_fn, get_profile
    from ragkit.retrieval import VectorIndex

    corpus = load_corpus(Path(corpus_path))
    questions, _ = filter_questions(load_questions(Path(questions_path)), {d["id"]: d for d in corpus})
    profile = get_profile(variant.model)

    start = time.perf_counter()
    embed_fn = create_embedding_fn(variant.model, backend=variant.backend, device=variant.device)
    index = VectorIndex.open(Path(index_path))
    load_s = time.perf_counter() - start

    row = measure(embed_fn, index, corpus, questions, format_query=profile.format_query,
                  load_s=load_s, n_latency=n_latency, peak_rss_fn=_peak_rss_mb)
    return {
        "variant": variant.name,
        "backend": variant.backend,
        "model": variant.model,
        "model_mb": model_size(variant.model, variant.backend) / 1e6,
        "install_mb": install_size(variant.deps) / 1e6,
        "n_questions": len(questions),
        **row,
    }


@app.command
def run(
    *,
    variants: list[str] | None = None,
    corpus: Path = Path("data/processed/law_docs.json"),
    questions: Path = Path("data/splits/test.jsonl"),
    out_dir: Path = Path("experiments/exp_008_deploy_bench/results"),
    n_latency: int = 100,
) -> None:
    """변형별로 측정하고 results.json과 comparison.md를 쓴다.

    Args:
        variants: 돌릴 변형 이름 (기본: torch-fp32 onnx-fp32 onnx-int8 onnx-int8-pruned).
        corpus: 코퍼스 JSON.
        questions: 평가 질문 (ragkit split의 test).
        out_dir: 결과 폴더.
        n_latency: 지연을 잴 질문 수.
    """
    chosen = [v for v in DEFAULT_VARIANTS if not variants or v.name in variants]
    missing = [v for v in chosen if not (Path(v.model) / ("onnx/model.onnx" if v.backend == "onnx" else "config.json")).exists()]
    if missing:
        names = ", ".join(v.name for v in missing)
        raise SystemExit(
            f"모델 폴더가 없습니다: {names}\n먼저 준비하세요:\n"
            "  uv run ragkit export-onnx models/multilingual-e5-small\n"
            "  uv run ragkit quantize models/multilingual-e5-small\n"
            "  uv run ragkit prune-vocab models/multilingual-e5-small\n"
            "  uv run ragkit export-onnx models/multilingual-e5-small-pruned\n"
            "  uv run ragkit quantize models/multilingual-e5-small-pruned"
        )

    rows = []
    ctx = mp.get_context("spawn")
    for variant in chosen:
        print(f"[{variant.name}] 인덱스 준비 (없으면 만든다)…", flush=True)
        with ctx.Pool(1) as pool:
            index_path = pool.apply(prepare_index, (variant, str(corpus)))
        print(f"[{variant.name}] 측정 중…", flush=True)
        with ctx.Pool(1) as pool:
            row = pool.apply(run_variant, (variant, str(corpus), str(questions), index_path, n_latency))
        rows.append(row)
        print(f"  지연 p50 {row['latency_ms_p50']:.1f}ms · R@5 {row['recall@5']:.3f} · 메모리 {row['peak_rss_mb']:.0f}MB")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"variants": [asdict(v) for v in chosen], "rows": rows}, ensure_ascii=False, indent=2)
    )
    table = to_markdown(rows)
    (out_dir / "comparison.md").write_text(f"# 배포 최적화 비교 (질문 {rows[0]['n_questions']}개, CPU)\n\n{table}\n")
    print("\n" + table)
    print(f"\n결과 → {out_dir}/comparison.md")
    log_to_mlflow(rows, out_dir, n_latency=n_latency)


def log_to_mlflow(rows: list[dict], out_dir: Path, *, n_latency: int) -> None:
    """MLflow가 켜져 있으면 벤치 한 번 = 부모 run, 변형마다 자식 run으로 남긴다 (없으면 아무 일도 안 함)."""
    from ragkit import tracking

    with tracking.run("deploy_bench", params={"n_questions": rows[0]["n_questions"], "n_latency": n_latency},
                      tags={"stage": "bench"}) as parent:
        for row in rows:
            params = {k: row.get(k) for k in ("variant", "backend", "model")}
            with tracking.run(row["variant"], params=params, nested=True) as run:
                run.log_metrics({k: v for k, v in row.items() if k not in params})
        parent.log_artifact(out_dir / "comparison.md")
        parent.log_artifact(out_dir / "results.json")


if __name__ == "__main__":
    app()
