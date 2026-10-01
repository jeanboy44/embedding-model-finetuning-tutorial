"""DS용 학습·평가 CLI: ragkit split | train | evaluate.

순서: ragkit split → ragkit train --config experiments/exp_00X/config.yaml → ragkit evaluate <폴더>
"""

import json
import time
from pathlib import Path
from typing import Literal

from ragkit import tracking
from ragkit.config import get_settings
from ragkit.data import filter_questions, load_corpus, load_questions
from ragkit.embeddings import choose_backend, create_embedding_fn, get_profile
from ragkit.rag.query_expansion import (
    ExpansionUnavailable,
    default_cache_path,
    expand_queries,
    expansion_summary,
    search_text,
)
from ragkit.training.split import (
    SPLITS,
    load_splits,
    split_by_law,
    split_summary,
    write_splits,
)


def _paths(corpus: Path | None, splits: Path | None) -> tuple[Path, Path]:
    data_dir = get_settings().data_dir
    return (
        corpus or data_dir / "processed" / "law_docs.json",
        splits or data_dir / "splits",
    )


def _fail(message: str) -> None:
    print(f"오류: {message}")
    raise SystemExit(1)


def _looks_like_path(model: str) -> bool:
    """HF 모델 이름(조직/이름, 슬래시 하나)이 아니라 로컬 경로로 쓴 값인지."""
    path = Path(model)
    return path.is_absolute() or model.startswith(".") or model.count("/") >= 2


def _load_corpus_or_fail(path: Path) -> list[dict]:
    if not path.exists():
        _fail(f"코퍼스가 없습니다: {path}\n  먼저 실행: uv run python scripts/prepare_law_data.py")
    return load_corpus(path)


def split(
    questions: Path,
    *,
    corpus: Path | None = None,
    out: Path | None = None,
    seed: int = 42,
    test_ratio: float = 0.2,
    dev_ratio: float = 0.1,
    strict: bool = False,
) -> None:
    """질문을 법령 단위로 train/dev/test에 나눈다.

    Args:
        questions: 질문 JSONL 파일 또는 폴더 (예: data/questions).
        corpus: 코퍼스 경로. 기본값은 data/processed/law_docs.json.
        out: 출력 폴더. 기본값은 data/splits.
        seed: 법령 섞기 seed.
        test_ratio: 테마(그룹)별 test 질문 비율.
        dev_ratio: 테마(그룹)별 dev 질문 비율.
        strict: 코퍼스에 없는 질문이 하나라도 있으면 중단한다.
    """
    corpus_path, out = _paths(corpus, out)
    if not Path(questions).exists():
        _fail(f"질문 파일이 없습니다: {questions}\n  law-question-gen 스킬로 먼저 만드세요.")
    docs = _load_corpus_or_fail(corpus_path)
    corpus_by_id = {doc["id"]: doc for doc in docs}

    kept, stats = filter_questions(load_questions(questions), corpus_by_id)
    print(
        f"질문 {len(kept)}개 사용 (코퍼스에 없어 뺀 질문 {stats['missing_positive']}개, "
        f"지운 negative {stats['dropped_negatives']}개)"
    )
    if strict and stats["missing_positive"]:
        _fail("--strict: 코퍼스에 없는 질문이 있습니다.")
    if not kept:
        _fail("남은 질문이 없습니다. 질문과 코퍼스가 같은 버전인지 확인하세요.")

    splits = split_by_law(kept, corpus_by_id, test_ratio=test_ratio, dev_ratio=dev_ratio, seed=seed)
    summary = split_summary(splits, corpus_by_id)
    meta = {
        "seed": seed,
        "test_ratio": test_ratio,
        "dev_ratio": dev_ratio,
        "questions": str(questions),
        "corpus": str(corpus_path),
        "filter": stats,
        **summary,
    }
    write_splits(splits, out, meta)
    for name, info in summary.items():
        print(f"  {name}: 질문 {info['questions']}개, 법령 {len(info['laws'])}개")
        if not info["questions"]:
            print(f"  경고: {name}이 비었습니다 (법령이 적은 테마).")
    print(f"완료 → {out}")


def train(
    config: Path,
    *,
    splits: Path | None = None,
    corpus: Path | None = None,
    model: str | None = None,
    max_steps: int | None = None,
    limit: int | None = None,
    dev_eval: bool = True,
) -> None:
    """실험 설정(config.yaml)대로 임베딩 모델을 파인튜닝한다. [train] extra 필요.

    Args:
        config: 실험 설정 경로 (예: experiments/exp_002_finetuned/config.yaml).
        splits: ragkit split 출력 폴더. 기본값은 data/splits.
        corpus: 코퍼스 경로. 기본값은 data/processed/law_docs.json.
        model: 설정의 base 모델 대신 쓸 모델 이름 또는 폴더.
        max_steps: 학습 step 수 제한 (강의에서 짧게 돌려 볼 때).
        limit: 학습 질문 수 제한.
        dev_eval: 학습 전·후 dev 평가(전체 코퍼스 임베딩 2회). --no-dev-eval로 끈다.
    """
    from ragkit.training import train as train_module

    corpus_path, splits_dir = _paths(corpus, splits)
    train_config = train_module.load_train_config(config)
    overrides = {
        "model": model,
        "max_steps": max_steps,
        "limit": limit,
        "dev_eval": None if dev_eval else False,
    }
    train_config = train_config.model_copy(
        update={k: v for k, v in overrides.items() if v is not None}
    )

    data = load_splits(splits_dir)
    if not data["train"]:
        _fail(f"train 분할이 없습니다: {splits_dir}\n  먼저 실행: ragkit split <질문 파일|폴더>")
    docs = _load_corpus_or_fail(corpus_path)
    # split 때와 다른 코퍼스일 수 있으므로 이 코퍼스 기준으로 한 번 더 거른다.
    corpus_by_id = {doc["id"]: doc for doc in docs}
    train_questions, train_stats = filter_questions(data["train"], corpus_by_id)
    dev_questions, dev_stats = filter_questions(data["dev"], corpus_by_id)
    missing = train_stats["missing_positive"] + dev_stats["missing_positive"]
    if missing:
        print(f"경고: 코퍼스에 없는 질문 {missing}개를 뺐습니다 (split 때와 코퍼스가 다름).")
    if not train_questions:
        _fail("이 코퍼스에 맞는 train 질문이 없습니다. 같은 코퍼스로 ragkit split을 다시 실행하세요.")

    settings = get_settings()
    params = _flat_params(train_config.model_dump(mode="json"))
    with tracking.run(f"train/{Path(train_config.output_dir).name}", params=params,
                      tags={"stage": "train", "config": str(config)}) as run:
        run.log_data({"corpus": corpus_path, "train": splits_dir / "train.jsonl", "dev": splits_dir / "dev.jsonl"})
        meta = train_module.train(
            train_config,
            train_questions,
            dev_questions,
            docs,
            query_prefix=settings.query_prefix,
            passage_prefix=settings.passage_prefix,
        )
        _log_train(run, meta, Path(train_config.output_dir))
    print(
        f"완료: {meta['train_examples']}개 예시, {meta['seconds']:.0f}초, "
        f"학습 파라미터 {meta['trainable_params']:,} / {meta['total_params']:,}"
        f" → {train_config.output_dir}"
    )


def evaluate(
    model: str,
    *,
    split: Literal["train", "dev", "test"] = "test",
    splits: Path | None = None,
    corpus: Path | None = None,
    backend: Literal["onnx", "torch", "st"] | None = None,
    index: Path | None = None,
    out: Path | None = None,
    expand: bool = False,
    expansion_cache: Path | None = None,
    rpm: float | None = None,
) -> None:
    """모델의 검색 성능을 전체 코퍼스 대상으로 잰다.

    코퍼스 임베딩은 SQLite 인덱스(ragkit.retrieval.build_index)에 저장해 두고,
    같은 모델·같은 코퍼스로 다시 평가하면 그 파일을 재사용한다(1단계 RAG 실습과 같은 파일).
    --expand면 질문을 LLM으로 법령 용어 검색어로 확장해(원문 + 확장어) 검색한다.

    Args:
        model: 모델 이름(base) 또는 학습한 모델 폴더.
        split: 평가할 분할 (test | dev | train).
        splits: ragkit split 출력 폴더. 기본값은 data/splits.
        corpus: 코퍼스 경로. 기본값은 data/processed/law_docs.json.
        backend: 임베딩 백엔드 (onnx | torch | st). 기본값은 모델 프로필이 정한다.
            onnx는 {모델 폴더}/onnx/model.onnx가 필요하고, 없으면 torch로 바꾼다.
        index: 인덱스 파일 경로. 기본값은 ragkit index와 같은 data/processed/index/{모델 키}.sqlite
            (학습한 폴더는 "exp_002-가중치수정시각", 허브 모델은 이름 끝부분).
        out: 결과 JSON 경로. 기본값은 experiments/results/{모델 이름}_{split}.json.
        expand: LLM 쿼리 확장을 쓴다(결과 파일 이름 끝에 _expand). 확장은 캐시에 있으면 재사용한다.
        expansion_cache: 확장 캐시 JSONL. 기본값은 data/processed/query_expansion/{LLM}.jsonl.
        rpm: LLM 분당 호출 한도 (무료 등급이면 15). 캐시에 없는 질문을 받을 때만 쓴다.
    """
    if split not in SPLITS:
        _fail(f"알 수 없는 분할: {split} (가능: {', '.join(SPLITS)})")
    corpus_path, splits_dir = _paths(corpus, splits)
    questions = load_splits(splits_dir)[split] if splits_dir.exists() else []
    if not questions:
        _fail(
            f"{split} 분할이 없거나 비었습니다: {splits_dir}\n"
            "  먼저 실행: ragkit split <질문 파일|폴더>"
        )
    docs = _load_corpus_or_fail(corpus_path)

    queries = [q["query"] for q in questions]
    expansions = _expansions(queries, expansion_cache, rpm) if expand else None
    result = _evaluate_model(
        model, docs, questions, backend=backend, index=index, expansions=expansions
    )
    result = {"split": split, **result}
    model_path = Path(model)

    suffix = "_expand" if expand else ""
    out = out or get_settings().results_dir / f"{model_path.name}_{split}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    with tracking.run(f"evaluate/{model_path.name}/{split}{suffix}", params=_eval_params(result, expand),
                      tags={"stage": "evaluate"}) as run:
        run.log_data({"corpus": corpus_path, split: splits_dir / f"{split}.jsonl"})
        _log_eval(run, result, out)

    print(f"{model}{' + 쿼리 확장' if expand else ''} · {split} ({result['n']}개 질문)")
    _print_metrics(result)
    print(f"결과 → {out}")


def expand(
    questions: Path,
    *,
    cache: Path | None = None,
    llm: str | None = None,
    rpm: float | None = None,
) -> None:
    """질문을 LLM으로 법령 용어 검색어로 확장해 캐시에 쌓는다 (evaluate --expand 전에 미리 받기).

    이미 캐시에 있는 질문은 건너뛰므로, 한도에 걸려 멈추면 같은 명령을 다시 실행하면 된다.

    Args:
        questions: 질문 JSONL 파일 또는 폴더 (예: data/splits/test.jsonl).
        cache: 확장 캐시 JSONL. 기본값은 data/processed/query_expansion/{LLM}.jsonl.
        llm: Gemini 모델 이름. 기본값은 설정의 gemini_model_name.
        rpm: 분당 호출 한도 (무료 등급이면 15).
    """
    if not Path(questions).exists():
        _fail(f"질문 파일이 없습니다: {questions}")
    queries = [q["query"] for q in load_questions(Path(questions))]
    llm = llm or get_settings().gemini_model_name
    cache = cache or default_cache_path(llm)
    print(f"질문 {len(queries)}개 확장 ({llm}) → {cache}")
    _expansions(queries, cache, rpm, llm)
    print("완료")


def _print_progress(done: int, total: int) -> None:
    if done == total or done % 20 == 0:
        print(f"  {done}/{total}")


def _expansions(
    queries: list[str], cache: Path | None, rpm: float | None, llm: str | None = None
) -> dict:
    """질문 → 확장 결과. 캐시에 없는 질문은 LLM으로 받는다."""
    from google.genai import errors

    llm = llm or get_settings().gemini_model_name
    cache = cache or default_cache_path(llm)
    try:
        return expand_queries(
            queries,
            cache,
            llm=llm,
            min_interval_s=60 / rpm if rpm else 0.0,
            on_progress=_print_progress,
        )
    except ExpansionUnavailable as e:
        _fail(str(e))
        raise
    except errors.APIError as e:
        if e.code != 429:
            raise
        _fail(
            f"LLM 호출 한도를 넘었습니다 ({llm}). 지금까지 받은 확장은 {cache}에 저장됐으니 "
            f"한도가 풀린 뒤 같은 명령을 다시 실행하세요.\n  {e.message}"
        )
        raise


def _print_metrics(result: dict) -> None:
    for judge in ("doc", "article"):
        metrics = "  ".join(f"{k} {v:.3f}" for k, v in result[judge].items())
        print(f"  [{judge}] {metrics}")


def _evaluate_model(
    model: str,
    docs: list[dict],
    questions: list[dict],
    *,
    backend: str | None,
    index: Path | None,
    expansions: dict | None = None,
) -> dict:
    """모델 하나를 모델 프로필의 입력 형식으로 인덱싱(또는 재사용)하고 평가한다.

    expansions(질문 → Expansion)를 주면 원문 + 확장어로 검색하고 확장 비용을 결과에 넣는다.
    """
    from ragkit.evaluation import evaluate_index
    from ragkit.retrieval import build_index, default_index_path, model_key

    model_path = Path(model)
    if not model_path.is_dir() and _looks_like_path(model):
        _fail(
            f"모델 폴더가 없습니다: {model}\n"
            "  먼저 실행: ragkit train --config experiments/<실험>/config.yaml"
        )
    checkpoint = model_path if model_path.is_dir() else None
    profile = get_profile(model)
    backend = backend or choose_backend(profile, None)
    onnx_file = model_path / "onnx" / "model.onnx"
    weights = model_path / "model.safetensors"
    if backend == "onnx" and checkpoint and not onnx_file.exists() and "torch" in profile.backends:
        print(f"참고: {onnx_file}가 없어 torch 백엔드로 평가합니다 (ragkit export-onnx로 만들 수 있음).")
        backend = "torch"
    if (
        backend == "onnx"
        and onnx_file.exists()
        and weights.exists()
        and onnx_file.stat().st_mtime < weights.stat().st_mtime
    ):
        _fail(
            f"ONNX 파일이 학습한 가중치보다 오래됐습니다: {onnx_file}\n"
            f"  먼저 실행: ragkit export-onnx {model_path}"
        )
    embed_fn = create_embedding_fn(model, checkpoint_path=checkpoint, backend=backend)
    key = model_key(model, checkpoint)
    index = index or default_index_path(key)
    start = time.perf_counter()
    vector_index = build_index(docs, embed_fn, index, model_key=key, format_doc=profile.format_doc)
    index_seconds = time.perf_counter() - start
    searched = questions
    if expansions:
        searched = [
            {**q, "query": search_text(q["query"], expansions[q["query"]].expansion)}
            for q in questions
        ]
    try:
        dim = vector_index.dim
        result = evaluate_index(
            vector_index, embed_fn, docs, searched, format_query=profile.format_query
        )
    finally:
        vector_index.close()
    if expansions:
        for row, question in zip(result["per_question"], questions):
            row["expanded_query"] = row["query"]
            row["query"] = question["query"]
        result["expansion"] = expansion_summary([expansions[q["query"]] for q in questions])
    return {
        "model": model,
        "backend": backend,
        "index": str(index),
        "index_seconds": index_seconds,  # 기존 인덱스를 재사용하면 몇 초
        "dim": dim,
        "deployable": profile.deployable,
        **result,
    }


def compare(
    config: Path,
    *,
    questions: Path | None = None,
    corpus: Path | None = None,
    backend: Literal["onnx", "torch", "st"] | None = None,
    index_dir: Path | None = None,
    out_dir: Path | None = None,
    expansion_cache: Path | None = None,
    rpm: float | None = None,
) -> None:
    """실험 설정의 모델들을 같은 질문·같은 코퍼스로 평가해 비교표를 만든다 (학습 없음).

    Args:
        config: 실험 설정 (예: experiments/exp_005_base_model_comparison/config.yaml).
            models(모델 이름 또는 폴더 목록), questions(질문 파일 또는 폴더)를 적는다.
            models 항목을 {model: 이름, expand: true}로 쓰면 LLM 쿼리 확장으로 평가한다.
        questions: 설정의 questions 대신 쓸 질문 파일 또는 폴더.
        corpus: 코퍼스 경로. 기본값은 data/processed/law_docs.json.
        backend: 모든 모델에 쓸 백엔드. 기본값은 모델마다 프로필이 정한다.
        index_dir: 인덱스를 둘 폴더. 기본값은 ragkit index와 같은 data/processed/index/.
        out_dir: 결과 폴더. 기본값은 {설정 폴더}/results/.
        expansion_cache: 쿼리 확장 캐시 JSONL. 기본값은 data/processed/query_expansion/{LLM}.jsonl.
        rpm: LLM 분당 호출 한도 (무료 등급이면 15). 캐시에 없는 질문을 받을 때만 쓴다.
    """
    import yaml


    settings = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    models = settings.get("models") or []
    if not models:
        _fail(f"비교할 모델이 없습니다: {config}의 models에 모델 이름을 적으세요.")
    questions_path = questions or settings.get("questions")
    if not questions_path or not Path(questions_path).exists():
        _fail(f"질문 파일이 없습니다: {questions_path}\n  먼저 실행: ragkit split <질문 파일|폴더>")
    corpus_path, _ = _paths(corpus, None)
    docs = _load_corpus_or_fail(corpus_path)
    kept, stats = filter_questions(load_questions(Path(questions_path)), {d["id"]: d for d in docs})
    if not kept:
        _fail("코퍼스에 맞는 질문이 없습니다.")
    print(f"질문 {len(kept)}개 × 모델 {len(models)}개 (코퍼스에 없어 뺀 질문 {stats['missing_positive']}개)")

    out_dir = out_dir or Path(config).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 비교 한 번 = 부모 run, 모델마다 자식 run (MLflow UI에서 자식끼리 지표를 나란히 비교)
    parent = tracking.run(
        settings.get("name") or Path(config).parent.name,
        params={"config": str(config), "questions": str(questions_path), "n": len(kept)},
        tags={"stage": "compare"},
    )
    with parent as parent_run:
        parent_run.log_data({"corpus": corpus_path, "questions": Path(questions_path)})
        rows = _compare_models(models, docs, kept, out_dir, backend, index_dir, expansion_cache, rpm)
        table = {
            "name": settings.get("name"),
            "questions": str(questions_path),
            "corpus": str(corpus_path),
            "n": len(kept),
            "models": rows,
        }
        (out_dir / "comparison.json").write_text(
            json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "comparison.md").write_text(_comparison_markdown(table), encoding="utf-8")
        parent_run.log_artifact(out_dir / "comparison.md")
        parent_run.log_artifact(out_dir / "comparison.json")
    print(f"비교표 → {out_dir / 'comparison.md'}")


def _compare_models(
    models: list,
    docs: list[dict],
    kept: list[dict],
    out_dir: Path,
    backend: str | None,
    index_dir: Path | None,
    expansion_cache: Path | None,
    rpm: float | None,
) -> list[dict]:
    """compare의 모델별 평가: 결과 파일 + 자식 run을 남기고 비교표 행을 돌려준다."""
    from ragkit.retrieval import model_key

    rows = []
    expansions = None
    for entry in models:
        model, expand = (entry["model"], bool(entry.get("expand"))) if isinstance(entry, dict) else (entry, False)
        print(f"=== {model}{' + 쿼리 확장' if expand else ''}")
        if expand and expansions is None:
            expansions = _expansions([q["query"] for q in kept], expansion_cache, rpm)

        model_path = Path(model)
        key = model_key(model, model_path if model_path.is_dir() else None)
        index = index_dir / f"{key}.sqlite" if index_dir else None
        result = _evaluate_model(
            model, docs, kept, backend=backend, index=index,
            expansions=expansions if expand else None,
        )
        result_path = out_dir / f"{key}{'_expand' if expand else ''}.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_metrics(result)
        row = {k: v for k, v in result.items() if k != "per_question"}
        with tracking.run(f"{key}{' + expand' if expand else ''}", params=_eval_params(result, expand),
                          nested=True) as run:
            _log_eval(run, result, result_path)
        rows.append(row)
    return rows


# ============================================================
# MLflow 기록 (MLFLOW_TRACKING_URI가 없으면 아무 일도 하지 않는다)
# ============================================================
def _flat_params(config: dict, prefix: str = "") -> dict:
    """중첩 설정(lora: {r: 8})을 MLflow 파라미터(lora.r)로 편다."""
    out = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out |= _flat_params(value, f"{prefix}{key}.")
        else:
            out[f"{prefix}{key}"] = value
    return out


def _log_train(run: tracking.Run, meta: dict, output_dir: Path) -> None:
    """학습 결과: 학습 전(step 0)·epoch별 dev 지표를 step으로, 시간·파라미터 수, train_meta.json."""
    if meta.get("dev_before"):
        run.log_metrics(meta["dev_before"], step=0)
    for row in meta.get("dev_history") or []:
        run.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=round(row["epoch"]))
    run.log_metrics({
        "train_seconds": meta.get("seconds"),
        "train_examples": meta.get("train_examples"),
        "trainable_params": meta.get("trainable_params"),
        "best_epoch": meta.get("best_epoch"),
    })
    run.log_artifact(output_dir / "train_meta.json")


def _eval_params(result: dict, expand: bool) -> dict:
    return {
        "model": result["model"],
        "split": result.get("split"),
        "backend": result.get("backend"),
        "dim": result.get("dim"),
        "n": result.get("n"),
        "expand": expand,
    }


def _log_eval(run: tracking.Run, result: dict, result_path: Path) -> None:
    """평가 결과: 지표(doc·article·유형별 R@5), 인덱싱 시간, 확장 비용, 결과 JSON."""
    run.log_metrics(tracking.flat_metrics(result))
    run.log_metrics({"index_seconds": result.get("index_seconds")})
    run.log_metrics({f"expansion/{k}": v for k, v in (result.get("expansion") or {}).items()})
    run.log_artifact(result_path)


def _row_label(row: dict) -> str:
    return f"{row['model']} + 쿼리 확장" if row.get("expansion") else row["model"]


def _comparison_markdown(table: dict) -> str:
    """비교표 Markdown. 판정은 doc(정답 문서 일치)이 기본, article(같은 조 조각이면 정답)은 참고."""
    lines = [
        f"# {table.get('name') or '모델 비교'}",
        "",
        f"질문 {table['n']}개, 코퍼스 `{table['corpus']}` 전체 대상 검색.",
        "",
        (
            "| 모델 | 백엔드 | 차원 | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | article R@10 "
            "| LLM 호출/질문 | 확장 지연 s | 배포 가능 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table["models"]:
        doc, article = row["doc"], row["article"]
        expansion = row.get("expansion")
        cost = (
            f"| {expansion['llm_calls_per_query']} | {expansion['latency_s_mean']:.2f} "
            if expansion else "| 0 | - "
        )
        lines.append(
            f"| {_row_label(row)} | {row['backend']} | {row['dim']} "
            f"| {doc.get('recall@1', 0):.3f} | {doc.get('recall@5', 0):.3f} "
            f"| {doc.get('recall@10', 0):.3f} | {doc['mrr@10']:.3f} | {doc['ndcg@10']:.3f} "
            f"| {article.get('recall@10', 0):.3f} {cost}| {'예' if row['deployable'] else '아니오'} |"
        )
    themes = sorted({theme for row in table["models"] for theme in row["by_theme"]})
    if themes:
        lines += ["", "## 테마별 R@10 (doc)", "", "| 모델 | " + " | ".join(themes) + " |",
                  "|---|" + "---|" * len(themes)]
        for row in table["models"]:
            cells = [
                f"{row['by_theme'][t]['doc'].get('recall@10', 0):.3f} (n={row['by_theme'][t]['n']})"
                if t in row["by_theme"] else "-"
                for t in themes
            ]
            lines.append(f"| {_row_label(row)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
