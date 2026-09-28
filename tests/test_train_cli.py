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
    first = train_cli._model_key(folder)
    os.utime(weights, (2_000, 2_000))

    assert first != train_cli._model_key(folder)
    assert train_cli._model_key(tmp_path / "multilingual-e5-small") == "multilingual-e5-small"


def test_evaluate_command_missing_splits(tmp_path) -> None:
    with pytest.raises(SystemExit):
        train_cli.evaluate("m", splits=tmp_path / "없음", corpus=tmp_path / "law_docs.json")
