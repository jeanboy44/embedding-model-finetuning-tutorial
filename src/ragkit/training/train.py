"""임베딩 파인튜닝: sentence-transformers + MultipleNegativesRankingLoss (전체 학습 / LoRA).

dev 질문이 있으면 epoch마다 dev R@5를 재고 가장 좋은 epoch의 가중치를 저장한다.
사람이 epoch를 고르지 않으므로 질문 데이터만 바꿔도 같은 규칙으로 재현된다.

[train] extra가 필요하다(uv sync --extra train). 무거운 import는 함수 안에서 한다.
저장 결과는 sentence-transformers 폴더라 ragkit torch 백엔드와 `ragkit export-onnx`가 그대로 읽는다.
LoRA는 학습 후 원래 가중치에 합쳐(merge) 같은 형식으로 저장한다.
"""

import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from ragkit.data import doc_text, questions_digest
from ragkit.training.triplets import to_examples

INSTALL_HINT = "학습에는 [train] extra가 필요합니다: uv sync --extra train"
# epoch 선택 기준. InformationRetrievalEvaluator(name="dev")가 내는 이름이다.
DEV_METRIC = "dev_cosine_recall@5"


class LoraSettings(BaseModel):
    """LoRA 설정. attention의 query/key/value에 저랭크 행렬을 더해 그것만 학습한다."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] = ["query", "key", "value"]
    save_adapter: bool = False


class TrainConfig(BaseModel):
    """학습 설정. 실험 config.yaml의 training: 블록과 같은 모양이다."""

    model: str = "intfloat/multilingual-e5-small"
    output_dir: Path
    batch_size: int = 32
    epochs: float = 1.0
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    num_negatives: int = 1
    # mnrl: 배치 전체를 한 번에 순전파. cached_mnrl: mini_batch_size씩 나눠 계산해(GradCache)
    # 손실·기울기는 같고 메모리만 줄인다. 큰 batch_size(in-batch negative 수)를 쓸 때 필요하다.
    loss: Literal["mnrl", "cached_mnrl"] = "mnrl"
    mini_batch_size: int = 8
    # attention dropout. None이면 모델 기본값(e5: 0.1). 0이면 sdpa 빠른 커널을 쓸 수 있다
    # (MPS의 sdpa는 dropout을 지원하지 않아 cached_mnrl의 기울기 없는 순전파가 실패한다).
    attention_dropout: float | None = None
    seed: int = 42
    max_steps: int | None = None
    limit: int | None = None
    # 학습 전·epoch마다 dev 평가와 best epoch 선택. epoch마다 전체 코퍼스를 임베딩하므로
    # 짧게 돌려 볼 때는 끈다 (끄면 마지막 epoch를 저장한다).
    dev_eval: bool = True
    # 복수 정답 판정 파일(JSONL). 주면 dev 평가가 alt_positive_ids도 정답으로 센다 (모델 선택 기준).
    dev_labels: Path | None = None
    # epoch마다 train 질문 n개로도 코퍼스 전체 검색을 잰다 (학습 데이터에 얼마나 맞춰졌는지, 0이면 끔).
    train_eval_sample: int = 0
    # 채굴한 hard negative(JSONL: qid, negative_ids). 주면 train 질문의 hard_negative_ids를 이것으로 바꾼다.
    negatives_file: Path | None = None
    lora: LoraSettings | None = None


def load_train_config(path: Path) -> TrainConfig:
    """실험 config.yaml의 training: 블록을 읽는다."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainConfig(**data["training"])


def _dev_evaluator(dev_questions, corpus, query_prefix, passage_prefix, name="dev"):
    """학습 중 평가기 (정답 = positive_id와 alt_positive_ids).

    ragkit.evaluation과 달리 related_ids·partial_ids를 순위에서 빼지 않고 article 판정도 없다.
    학습이 좋아지는지 방향만 보는 용도이고, 최종 점수는 ragkit evaluate로 낸다.
    """
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    return InformationRetrievalEvaluator(
        queries={str(i): query_prefix + q["query"] for i, q in enumerate(dev_questions)},
        corpus={doc["id"]: passage_prefix + doc_text(doc) for doc in corpus},
        relevant_docs={
            str(i): {q["positive_id"], *(q.get("alt_positive_ids") or [])} for i, q in enumerate(dev_questions)
        },
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        map_at_k=[10],
        name=name,
        write_csv=False,
        batch_size=64,
    )


def _best_epoch_callback(model, history: list[dict], best: dict):
    """epoch마다 dev 지표를 history에 쌓고, DEV_METRIC이 가장 좋은 때의 가중치를 best에 둔다.

    체크포인트 파일 대신 CPU 메모리에 복사해 둔다 (e5-small은 약 0.5GB).
    LoRA는 병합 전 모델 전체의 state_dict라 같은 모델에 되돌려 넣을 수 있다.
    """
    from transformers import TrainerCallback

    class BestEpoch(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            dev = {
                k.removeprefix("eval_"): v
                for k, v in (metrics or {}).items()
                if k.startswith(("eval_dev_", "eval_train_"))
            }
            if DEV_METRIC not in dev:
                return
            history.append({"epoch": state.epoch, **dev})
            if "score" not in best or dev[DEV_METRIC] > best["score"]:
                best.update(
                    score=dev[DEV_METRIC],
                    epoch=round(state.epoch),
                    metrics=dev,
                    state={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
                )

    return BestEpoch()


def _make_loss(config: TrainConfig, model):
    from sentence_transformers.sentence_transformer import losses

    if config.loss == "cached_mnrl":
        return losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=config.mini_batch_size)
    return losses.MultipleNegativesRankingLoss(model)


def train(
    config: TrainConfig,
    train_questions: list[dict],
    dev_questions: list[dict],
    corpus: list[dict],
    *,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
) -> dict:
    """모델을 학습하고 config.output_dir에 저장한다.

    Args:
        config: 학습 설정.
        train_questions: filter_questions를 거친 학습 질문.
        dev_questions: 학습 전·후 점수를 볼 dev 질문. 비어 있으면 건너뛴다.
        corpus: 전체 코퍼스 (dev 평가 대상, 문서 텍스트 조회).
        query_prefix: 질문 앞 문구.
        passage_prefix: 문서 앞 문구.

    Returns:
        train_meta.json에 저장한 내용. dev 평가를 했으면 dev_history(epoch별 지표),
        best_epoch, dev_after(= best epoch의 지표)가 들어 있다.

    Raises:
        ImportError: [train] extra가 없을 때.
        ValueError: 학습할 예시가 없을 때.
    """
    try:
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.sentence_transformer.training_args import (
            BatchSamplers,
        )
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    from ragkit.models import resolve_model_source

    corpus_by_id = {doc["id"]: doc for doc in corpus}
    questions = train_questions[: config.limit] if config.limit else train_questions
    examples = to_examples(
        questions,
        corpus_by_id,
        num_negatives=config.num_negatives,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    if not examples:
        raise ValueError("학습할 예시가 없습니다. hard negative가 있는 train 질문이 필요합니다.")

    model_source = config.model if Path(config.model).exists() else resolve_model_source(config.model)
    config_kwargs = (
        {"attention_probs_dropout_prob": config.attention_dropout}
        if config.attention_dropout is not None
        else None
    )
    model = SentenceTransformer(model_source, config_kwargs=config_kwargs)
    model.max_seq_length = config.max_seq_length
    total_params = sum(p.numel() for p in model.parameters())

    transformer: Any = model[0]  # sentence-transformers Transformer 모듈
    peft_model: Any = None
    if config.lora:
        from peft import LoraConfig, get_peft_model

        # auto_model은 읽기 전용 별칭이다. forward가 쓰는 실제 모듈(model)을 감싸야 LoRA가 적용된다.
        peft_model = get_peft_model(
            transformer.model,
            LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
            ),
        )
        transformer.model = peft_model
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    evaluator = (
        _dev_evaluator(dev_questions, corpus, query_prefix, passage_prefix)
        if dev_questions and config.dev_eval
        else None
    )
    if evaluator and config.train_eval_sample:
        from sentence_transformers.base.evaluation.sequential import SequentialEvaluator

        sample = random.Random(config.seed).sample(questions, min(config.train_eval_sample, len(questions)))
        evaluator = SequentialEvaluator(
            [evaluator, _dev_evaluator(sample, corpus, query_prefix, passage_prefix, name="train")]
        )
    dev_before = evaluator(model) if evaluator else None
    history: list[dict] = []
    best: dict = {}

    output_dir = Path(config.output_dir)
    # 예전 LoRA 학습이 남긴 어댑터가 새 모델과 섞이지 않게 지운다.
    shutil.rmtree(output_dir / "adapter", ignore_errors=True)
    # Trainer 작업 폴더(save_strategy="no"라 비어 있음)는 모델 폴더 밖 임시 폴더에 둔다.
    work_dir = tempfile.TemporaryDirectory(prefix="ragkit-train-")
    args = SentenceTransformerTrainingArguments(
        output_dir=work_dir.name,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps or -1,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        warmup_steps=config.warmup_ratio,  # transformers 5: 1 미만 실수는 전체 step 대비 비율
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # 같은 문서가 한 배치에 두 번 → 가짜 negative 방지
        seed=config.seed,
        eval_strategy="epoch" if evaluator else "no",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(examples),
        loss=_make_loss(config, model),
        evaluator=evaluator,
        callbacks=[_best_epoch_callback(model, history, best)] if evaluator else None,
    )
    start = time.perf_counter()
    with work_dir:
        trainer.train()
    seconds = time.perf_counter() - start
    if best:
        model.load_state_dict(best["state"])
        dev_after = best["metrics"]
    else:
        # max_steps가 1 epoch보다 짧으면 epoch 끝 평가가 없다 → 마지막 가중치를 한 번 잰다
        dev_after = evaluator(model) if evaluator else None

    if config.lora and peft_model is not None:
        if config.lora.save_adapter:
            peft_model.save_pretrained(str(output_dir / "adapter"))
        transformer.model = peft_model.merge_and_unload()
    model.save(str(output_dir))

    meta = {
        "seconds": seconds,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_examples": len(examples),
        "dev_before": dev_before,
        "dev_after": dev_after,
        "dev_history": history,
        "best_epoch": best.get("epoch"),
        "data": {
            "train": {"n": len(questions), "digest": questions_digest(questions)},
            "dev": {"n": len(dev_questions), "digest": questions_digest(dev_questions)},
        },
        "config": config.model_dump(mode="json"),
    }
    (output_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
