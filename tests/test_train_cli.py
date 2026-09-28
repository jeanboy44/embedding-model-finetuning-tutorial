"""학습 CLI 테스트 (모델 없이 도는 부분)."""

import json
import os

import numpy as np
import pytest

from ragkit.cli import train_cli


def _write_inputs(tmp_path, corpus, questions):
    corpus_path = tmp_path / "law_docs.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    qpath = tmp_path / "questions.jsonl"
    extra = {"query": "옛 코퍼스", "positive_id": "없는법_법률_제1조", "hard_negative_ids": []}
    lines = [json.dumps(q, ensure_ascii=False) for q in [*questions, extra]]
    qpath.write_text("\n".join(lines), encoding="utf-8")
    return corpus_path, qpath


def test_split_command_writes_splits(tmp_path, corpus, questions, capsys) -> None:
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)

    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")

    meta = json.loads((tmp_path / "splits" / "split_meta.json").read_text(encoding="utf-8"))
    assert meta["train"]["questions"] + meta["dev"]["questions"] + meta["test"]["questions"] == 28
    assert meta["filter"]["missing_positive"] == 1
    assert "뺀 질문 1개" in capsys.readouterr().out


def test_split_command_strict_fails(tmp_path, corpus, questions) -> None:
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    with pytest.raises(SystemExit):
        train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits", strict=True)


def test_evaluate_command_writes_result(tmp_path, corpus, questions, monkeypatch) -> None:
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")

    def fake_create(model_name, checkpoint_path=None, device=None, backend=None):
        return lambda texts, batch_size=None: np.ones((len(texts), 2)) / np.sqrt(2)

    monkeypatch.setattr(train_cli, "create_embedding_fn", fake_create)
    out = tmp_path / "result.json"
    index_path = tmp_path / "index" / "base-model.sqlite"

    train_cli.evaluate(
        "base-model", splits=tmp_path / "splits", corpus=corpus_path, index=index_path, out=out
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["n"] == 8
    assert result["model"] == "base-model"
    assert result["split"] == "test"
    assert result["index"] == str(index_path)
    assert index_path.exists()  # 다음 평가 때 재사용된다


def test_model_key_changes_when_retrained(tmp_path) -> None:
    folder = tmp_path / "exp_002"
    folder.mkdir()
    weights = folder / "model.safetensors"
    weights.write_bytes(b"v1")
    os.utime(weights, (1_000, 1_000))
    first = train_cli._model_key(folder, "torch")
    os.utime(weights, (2_000, 2_000))

    assert first != train_cli._model_key(folder, "torch")
    assert train_cli._model_key(tmp_path / "없는폴더" / "e5", "torch") == "e5"


def test_model_key_depends_on_backend(tmp_path) -> None:
    """onnx 백엔드는 onnx 파일 기준 키라 torch로 만든 인덱스를 재사용하지 않는다."""
    folder = tmp_path / "exp_002"
    (folder / "onnx").mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(b"w")
    (folder / "onnx" / "model.onnx").write_bytes(b"o")
    os.utime(folder / "model.safetensors", (1_000, 1_000))
    os.utime(folder / "onnx" / "model.onnx", (1_000, 1_000))

    assert train_cli._model_key(folder, "torch") != train_cli._model_key(folder, "onnx")


def test_evaluate_default_index_keeps_shared_file(
    tmp_path, corpus, questions, monkeypatch
) -> None:
    """기본 인덱스 경로는 모델 키 이름이라, ragkit index가 만든 <폴더 이름>.sqlite를 지우지 않는다."""
    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    model_dir = tmp_path / "models" / "finetuned" / "exp_002"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"w")
    shared = tmp_path / "data" / "processed" / "index" / "exp_002.sqlite"
    shared.parent.mkdir(parents=True)
    shared.write_bytes("ragkit index가 만든 파일".encode())

    def fake_create(model_name, checkpoint_path=None, device=None, backend=None):
        return lambda texts, batch_size=None: np.ones((len(texts), 2)) / np.sqrt(2)

    monkeypatch.setattr(train_cli, "create_embedding_fn", fake_create)
    out = tmp_path / "result.json"

    train_cli.evaluate(str(model_dir), splits=tmp_path / "splits", corpus=corpus_path, out=out)

    result = json.loads(out.read_text(encoding="utf-8"))
    key = train_cli._model_key(model_dir, "torch")
    assert result["index"] == str(shared.parent / f"{key}.sqlite")
    assert shared.read_bytes() == "ragkit index가 만든 파일".encode()


@pytest.mark.parametrize("model", ["models/finetuned/exp_002", "./exp_002", "/없는/경로/exp"])
def test_evaluate_missing_model_folder_fails(tmp_path, corpus, questions, monkeypatch, model):
    """학습 전에 평가부터 실행하면 안내하고 종료한다 (HF 모델 이름으로 해석하지 않는다)."""
    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")

    def must_not_load(*args, **kwargs):
        raise AssertionError("모델을 불러오면 안 된다")

    monkeypatch.setattr(train_cli, "create_embedding_fn", must_not_load)
    with pytest.raises(SystemExit):
        train_cli.evaluate(model, splits=tmp_path / "splits", corpus=corpus_path)


def test_train_command_can_skip_dev_eval(tmp_path, corpus, questions, monkeypatch) -> None:
    """--no-dev-eval이면 학습 설정의 dev_eval이 꺼진다."""
    import ragkit.training.train as train_module

    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  output_dir: out\n", encoding="utf-8")
    seen = {}

    def fake_train(config, train_q, dev_q, docs, **kwargs):
        seen["dev_eval"] = config.dev_eval
        return {"train_examples": 1, "seconds": 0.0, "trainable_params": 1, "total_params": 1}

    monkeypatch.setattr(train_module, "train", fake_train)

    train_cli.train(
        config_path, splits=tmp_path / "splits", corpus=corpus_path, dev_eval=False
    )

    assert seen == {"dev_eval": False}


def test_evaluate_command_missing_splits(tmp_path) -> None:
    with pytest.raises(SystemExit):
        train_cli.evaluate("m", splits=tmp_path / "없음", corpus=tmp_path / "law_docs.json")


def _fake_create(model_name, checkpoint_path=None, device=None, backend=None):
    return lambda texts, batch_size=None: np.ones((len(texts), 2)) / np.sqrt(2)


def test_evaluate_rejects_unknown_split(tmp_path, corpus, questions, monkeypatch) -> None:
    """--split 오타는 KeyError가 아니라 안내 후 종료."""
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)
    with pytest.raises(SystemExit):
        train_cli.evaluate("m", split="tset", splits=tmp_path / "splits", corpus=corpus_path)


def test_evaluate_onnx_fails_when_onnx_older_than_weights(
    tmp_path, corpus, questions, monkeypatch
) -> None:
    """다시 학습한 뒤 ONNX를 다시 변환하지 않았으면 예전 모델로 평가하지 않고 안내한다."""
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    model_dir = tmp_path / "exp_002"
    (model_dir / "onnx").mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"w")
    (model_dir / "onnx" / "model.onnx").write_bytes(b"o")
    os.utime(model_dir / "onnx" / "model.onnx", (1_000, 1_000))
    os.utime(model_dir / "model.safetensors", (2_000, 2_000))
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)

    with pytest.raises(SystemExit):
        train_cli.evaluate(
            str(model_dir), backend="onnx", splits=tmp_path / "splits", corpus=corpus_path
        )


def test_train_command_refilters_questions_for_corpus(
    tmp_path, corpus, questions, monkeypatch
) -> None:
    """split 때와 다른 코퍼스를 주면 없는 질문을 빼고 학습한다 (KeyError 없이)."""
    import ragkit.training.train as train_module

    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    smaller = [d for d in corpus if not d["id"].startswith(("가법_", "나법_", "다법_", "라법_"))]
    small_path = tmp_path / "small.json"
    small_path.write_text(json.dumps(smaller, ensure_ascii=False), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  output_dir: out\n", encoding="utf-8")
    seen = {}

    def fake_train(config, train_q, dev_q, docs, **kwargs):
        ids = {d["id"] for d in docs}
        seen["ok"] = all(q["positive_id"] in ids for q in [*train_q, *dev_q])
        return {"train_examples": 1, "seconds": 0.0, "trainable_params": 1, "total_params": 1}

    monkeypatch.setattr(train_module, "train", fake_train)

    train_cli.train(config_path, splits=tmp_path / "splits", corpus=small_path)

    assert seen == {"ok": True}


def test_cli_help_shows_placeholders(capsys) -> None:
    """도움말에서 {모델 키}·{split} 같은 자리 표시가 rich markup으로 사라지지 않는다."""
    from ragkit.cli.cli_tool import app

    with pytest.raises(SystemExit):
        app(["evaluate", "--help"])

    out = capsys.readouterr().out  # 긴 설명은 줄바꿈되므로 줄 안에 남는 조각만 확인한다
    assert "_{split}.json" in out
    assert "}/onnx/model.onnx" in out
