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
    if use_lora:
        assert meta["trainable_params"] < meta["total_params"] / 10
        assert (out / "adapter" / "adapter_config.json").exists()
    else:
        assert meta["trainable_params"] == meta["total_params"]

    with safe_open(str(out / "model.safetensors"), "pt") as weights:
        saved_keys = list(weights.keys())  # safe_open은 dict가 아니라 keys()로만 이름을 준다
    assert saved_keys and not any("lora" in key for key in saved_keys)

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
