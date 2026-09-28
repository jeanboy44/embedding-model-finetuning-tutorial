"""임베딩 파인튜닝: sentence-transformers + MultipleNegativesRankingLoss (전체 학습 / LoRA).

[train] extra가 필요하다(uv sync --extra train). 무거운 import는 함수 안에서 한다.
저장 결과는 sentence-transformers 폴더라 ragkit torch 백엔드와 `ragkit export-onnx`가 그대로 읽는다.
LoRA는 학습 후 원래 가중치에 합쳐(merge) 같은 형식으로 저장한다.
"""

import json
import time
from pathlib import Path

from pydantic import BaseModel

from ragkit.data import doc_text
from ragkit.training.triplets import to_examples

INSTALL_HINT = "학습에는 [train] extra가 필요합니다: uv sync --extra train"


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
    seed: int = 42
    max_steps: int | None = None
    limit: int | None = None
    # 학습 전·후 dev 평가. 전체 코퍼스를 두 번 임베딩하므로 짧게 돌려 볼 때는 끈다.
    dev_eval: bool = True
    lora: LoraSettings | None = None


def load_train_config(path: Path) -> TrainConfig:
    """실험 config.yaml의 training: 블록을 읽는다."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainConfig(**data["training"])


def _dev_evaluator(dev_questions, corpus, query_prefix, passage_prefix):
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    return InformationRetrievalEvaluator(
        queries={str(i): query_prefix + q["query"] for i, q in enumerate(dev_questions)},
        corpus={doc["id"]: passage_prefix + doc_text(doc) for doc in corpus},
        relevant_docs={str(i): {q["positive_id"]} for i, q in enumerate(dev_questions)},
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        map_at_k=[10],
        name="dev",
        write_csv=False,
        batch_size=64,
    )


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
        train_meta.json에 저장한 내용.

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
        from sentence_transformers.sentence_transformer import losses
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
    model = SentenceTransformer(model_source)
    model.max_seq_length = config.max_seq_length
    total_params = sum(p.numel() for p in model.parameters())

    if config.lora:
        from peft import LoraConfig, get_peft_model

        # auto_model은 읽기 전용 별칭이다. forward가 쓰는 실제 모듈(model)을 감싸야 LoRA가 적용된다.
        model[0].model = get_peft_model(
            model[0].model,
            LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
            ),
        )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    evaluator = (
        _dev_evaluator(dev_questions, corpus, query_prefix, passage_prefix)
        if dev_questions and config.dev_eval
        else None
    )
    dev_before = evaluator(model) if evaluator else None

    output_dir = Path(config.output_dir)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps or -1,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        warmup_steps=config.warmup_ratio,  # transformers 5: 1 미만 실수는 전체 step 대비 비율
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # 같은 문서가 한 배치에 두 번 → 가짜 negative 방지
        seed=config.seed,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(examples),
        loss=losses.MultipleNegativesRankingLoss(model),
    )
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    dev_after = evaluator(model) if evaluator else None

    if config.lora:
        if config.lora.save_adapter:
            model[0].model.save_pretrained(str(output_dir / "adapter"))
        model[0].model = model[0].model.merge_and_unload()
    model.save(str(output_dir))

    meta = {
        "seconds": seconds,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_examples": len(examples),
        "dev_before": dev_before,
        "dev_after": dev_after,
        "config": config.model_dump(mode="json"),
    }
    (output_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
