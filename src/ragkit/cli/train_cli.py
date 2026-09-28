"""DS용 학습·평가 CLI: ragkit split | train | evaluate.

순서: ragkit split → ragkit train --config experiments/exp_00X/config.yaml → ragkit evaluate --model <폴더>
"""

import json
from pathlib import Path
from typing import Literal

from ragkit.config import get_settings
from ragkit.data import filter_questions, load_corpus, load_questions
from ragkit.embeddings import create_embedding_fn
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


def _model_key(model_path: Path, backend: str) -> str:
    """인덱스 재사용 판단에 쓰는 모델 키. 기본 인덱스 파일 이름으로도 쓴다.

    학습한 모델 폴더는 다시 학습해도 이름이 같으므로, 백엔드가 읽는 가중치 파일
    (torch: model.safetensors, onnx: onnx/model.onnx)의 수정 시각을 붙여 예전 모델이나
    다른 백엔드로 만든 인덱스를 재사용하지 않게 한다. 모델 이름(HF)이면 이름만 쓴다.
    """
    weights = model_path / ("onnx/model.onnx" if backend == "onnx" else "model.safetensors")
    if weights.exists():
        return f"{model_path.name}@{backend}-{int(weights.stat().st_mtime)}"
    return model_path.name


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
    meta = train_module.train(
        train_config,
        train_questions,
        dev_questions,
        docs,
        query_prefix=settings.query_prefix,
        passage_prefix=settings.passage_prefix,
    )
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
    backend: Literal["torch", "onnx"] = "torch",
    index: Path | None = None,
    out: Path | None = None,
) -> None:
    """모델의 검색 성능을 전체 코퍼스 대상으로 잰다.

    코퍼스 임베딩은 SQLite 인덱스(ragkit.retrieval.build_index)에 저장해 두고,
    같은 모델·같은 코퍼스로 다시 평가하면 그 파일을 재사용한다(1단계 RAG 실습과 같은 파일).

    Args:
        model: 모델 이름(base) 또는 학습한 모델 폴더.
        split: 평가할 분할 (test | dev | train).
        splits: ragkit split 출력 폴더. 기본값은 data/splits.
        corpus: 코퍼스 경로. 기본값은 data/processed/law_docs.json.
        backend: 임베딩 백엔드 (torch | onnx). onnx는 {모델 폴더}/onnx/model.onnx가 필요하다.
        index: 인덱스 파일 경로. 기본값은 data/processed/index/{모델 키}.sqlite
            (학습한 폴더는 "exp_002@torch-수정시각", HF 모델 이름은 이름 그대로).
        out: 결과 JSON 경로. 기본값은 experiments/results/{모델 이름}_{split}.json.
    """
    from ragkit.evaluation import evaluate_index
    from ragkit.retrieval import build_index

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

    model_path = Path(model)
    if not model_path.is_dir() and _looks_like_path(model):
        _fail(
            f"모델 폴더가 없습니다: {model}\n"
            "  먼저 실행: ragkit train --config experiments/<실험>/config.yaml"
        )
    onnx_file = model_path / "onnx" / "model.onnx"
    weights = model_path / "model.safetensors"
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
    checkpoint = model_path if model_path.is_dir() else None
    embed_fn = create_embedding_fn(model, checkpoint_path=checkpoint, backend=backend)
    settings = get_settings()
    key = _model_key(model_path, backend)
    index = index or settings.data_dir / "processed" / "index" / f"{key}.sqlite"
    vector_index = build_index(docs, embed_fn, index, model_key=key)
    try:
        result = evaluate_index(
            vector_index, embed_fn, docs, questions, query_prefix=settings.query_prefix
        )
    finally:
        vector_index.close()
    result = {"model": model, "split": split, "backend": backend, "index": str(index), **result}

    out = out or settings.results_dir / f"{model_path.name}_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{model} · {split} ({result['n']}개 질문)")
    for judge in ("doc", "article"):
        metrics = "  ".join(f"{k} {v:.3f}" for k, v in result[judge].items())
        print(f"  [{judge}] {metrics}")
    print(f"결과 → {out}")
