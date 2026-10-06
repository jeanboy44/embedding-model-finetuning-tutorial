"""전체 학습 / LoRA smoke 테스트: 몇 step 학습 → 저장 → ragkit torch 백엔드로 다시 읽어 평가."""

import json
import os
from pathlib import Path

import pytest

MODEL = Path(os.environ.get("RAGKIT_TEST_MODEL", "models/multilingual-e5-small"))

needs_model = pytest.mark.skipif(
    not (MODEL / "config.json").exists(),
    reason=f"로컬 모델 없음: {MODEL} (RAGKIT_TEST_MODEL로 지정)",
)


@pytest.mark.slow
@needs_model
@pytest.mark.parametrize("use_lora", [False, True], ids=["full", "lora"])
def test_train_saves_loadable_model(tmp_path, corpus, questions, use_lora) -> None:
    from safetensors import safe_open

    from ragkit.embeddings import create_embedding_fn
    from ragkit.evaluation import evaluate_retrieval
    from ragkit.training.train import LoraSettings, TrainConfig, train

    out = tmp_path / "model"
    # 같은 폴더에 예전 LoRA 학습이 남긴 어댑터가 있어도 새 학습 결과와 섞이지 않아야 한다
    (out / "adapter").mkdir(parents=True)
    (out / "adapter" / "stale.txt").write_text("예전 어댑터", encoding="utf-8")
    config = TrainConfig(
        model=str(MODEL),
        output_dir=out,
        batch_size=4,
        max_steps=2,
        max_seq_length=64,
        lora=LoraSettings(r=4, alpha=8, save_adapter=True) if use_lora else None,
    )

    meta = train(config, questions[:16], questions[16:20], corpus)

    assert json.loads((out / "train_meta.json").read_text(encoding="utf-8"))["seconds"] > 0
    assert meta["train_examples"] == 16
    assert meta["dev_before"] and meta["dev_after"]
    assert not (out / "checkpoints").exists()
    assert not (out / "adapter" / "stale.txt").exists()
    if use_lora:
        assert meta["trainable_params"] < meta["total_params"] / 10
        assert (out / "adapter" / "adapter_config.json").exists()
    else:
        assert meta["trainable_params"] == meta["total_params"]
        assert not (out / "adapter").exists()

    with safe_open(str(out / "model.safetensors"), "pt") as weights:
        saved_keys = list(weights.keys())  # safe_open은 dict가 아니라 keys()로만 이름을 준다
    assert saved_keys and not any("lora" in key for key in saved_keys)

    # 병합이 잘못돼 가중치 이름이 어긋나면 from_pretrained가 무작위 초기화로 채운다 → 빈 목록이어야 한다
    from transformers import AutoModel

    _, info = AutoModel.from_pretrained(str(out), output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]

    embed = create_embedding_fn(str(out), checkpoint_path=out, backend="torch", device="cpu")
    result = evaluate_retrieval(embed, corpus, questions[20:24])
    assert result["n"] == 4


def test_load_train_config(tmp_path) -> None:
    from ragkit.training.train import load_train_config

    path = tmp_path / "config.yaml"
    path.write_text(
        "name: lora\ntraining:\n  output_dir: models/finetuned/x\n  lora:\n    r: 8\n",
        encoding="utf-8",
    )

    config = load_train_config(path)

    assert config.output_dir == Path("models/finetuned/x")
    assert config.lora is not None and config.lora.r == 8 and config.lora.alpha == 32
    assert config.batch_size == 32
    assert config.loss == "mnrl"
    assert config.train_eval_sample == 0 and config.dev_labels is None


def test_load_train_config_cached_mnrl(tmp_path) -> None:
    from ragkit.training.train import load_train_config

    path = tmp_path / "config.yaml"
    path.write_text(
        "training:\n  output_dir: out\n  batch_size: 128\n  loss: cached_mnrl\n  mini_batch_size: 8\n",
        encoding="utf-8",
    )

    config = load_train_config(path)

    assert (config.loss, config.batch_size, config.mini_batch_size) == ("cached_mnrl", 128, 8)


@pytest.mark.slow
@needs_model
@pytest.mark.parametrize("use_lora", [False, True], ids=["full", "lora"])
def test_train_keeps_best_dev_epoch(tmp_path, corpus, questions, use_lora) -> None:
    """epoch마다 dev R@5를 재고, 가장 좋은 epoch의 가중치를 저장한다 (CachedMNRL)."""
    from ragkit.training.train import DEV_METRIC, LoraSettings, TrainConfig, train

    config = TrainConfig(
        model=str(MODEL),
        output_dir=tmp_path / "model",
        batch_size=4,
        epochs=3,
        lr=1e-4,
        max_seq_length=64,
        loss="cached_mnrl",
        mini_batch_size=2,
        attention_dropout=0.0,  # MPS sdpa는 dropout 미지원
        lora=LoraSettings(r=4, alpha=8) if use_lora else None,
    )

    meta = train(config, questions[:8], questions[16:20], corpus)

    history = meta["dev_history"]
    assert [round(h["epoch"]) for h in history] == [1, 2, 3]
    best = max(history, key=lambda h: h[DEV_METRIC])  # 동점이면 앞 epoch
    assert meta["best_epoch"] == round(best["epoch"])
    assert meta["dev_after"][DEV_METRIC] == best[DEV_METRIC]
    assert (tmp_path / "model" / "model.safetensors").exists()


def test_dev_evaluator_accepts_alt_positives(corpus) -> None:
    """학습 중 dev 평가도 복수 정답(alt_positive_ids)을 정답으로 센다."""
    from ragkit.training.train import _dev_evaluator

    ids = [d["id"] for d in corpus[:2]]
    questions = [{"query": "q", "positive_id": ids[0], "alt_positive_ids": [ids[1]]}]

    evaluator = _dev_evaluator(questions, corpus, "query: ", "passage: ")

    assert evaluator.relevant_docs == {"0": set(ids)}


@pytest.mark.slow
@needs_model
def test_train_records_train_sample_recall(tmp_path, corpus, questions) -> None:
    """train_eval_sample을 주면 epoch마다 train 질문 표본의 코퍼스 전체 검색 지표도 기록한다."""
    from ragkit.training.train import DEV_METRIC, TrainConfig, train

    config = TrainConfig(
        model=str(MODEL), output_dir=tmp_path / "model", batch_size=4, epochs=2, max_seq_length=64,
        loss="cached_mnrl", mini_batch_size=2, attention_dropout=0.0, train_eval_sample=4,
    )

    meta = train(config, questions[:8], questions[16:20], corpus)

    assert [round(h["epoch"]) for h in meta["dev_history"]] == [1, 2]
    assert all(DEV_METRIC in h and "train_cosine_recall@5" in h for h in meta["dev_history"])
    assert "train_cosine_recall@5" in meta["dev_before"]
