"""학습 CLI 테스트 (모델 없이 도는 부분)."""

import json
import os
from pathlib import Path

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


def test_evaluate_default_index_shared_with_ragkit_index(
    tmp_path, corpus, questions, monkeypatch
) -> None:
    """기본 인덱스 경로는 ragkit index와 같은 규칙(model_key, default_index_path)이라 파일을 공유한다."""
    from ragkit.retrieval import default_index_path, model_key

    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    model_dir = tmp_path / "models" / "finetuned" / "exp_002"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"w")
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)
    out = tmp_path / "result.json"

    train_cli.evaluate(str(model_dir), splits=tmp_path / "splits", corpus=corpus_path, out=out)

    result = json.loads(out.read_text(encoding="utf-8"))
    expected = default_index_path(model_key(str(model_dir), model_dir))
    assert Path(result["index"]).resolve() == expected.resolve()


def test_evaluate_applies_model_profile(tmp_path, corpus, questions, monkeypatch) -> None:
    """모델마다 입력 형식이 다르다: EmbeddingGemma는 task/title 형식으로 임베딩한다."""
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    seen: list[str] = []

    def recording_create(model_name, checkpoint_path=None, device=None, backend=None):
        def embed(texts, batch_size=None):
            seen.extend(texts)
            return np.ones((len(texts), 2)) / np.sqrt(2)

        return embed

    monkeypatch.setattr(train_cli, "create_embedding_fn", recording_create)

    train_cli.evaluate(
        "google/embeddinggemma-300m",
        splits=tmp_path / "splits",
        corpus=corpus_path,
        index=tmp_path / "g.sqlite",
        out=tmp_path / "g.json",
    )

    assert any(t.startswith("title: ") for t in seen)
    assert any(t.startswith("task: search result | query: ") for t in seen)
    assert not any(t.startswith(("query: ", "passage: ")) for t in seen)


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
    monkeypatch.chdir(tmp_path)  # 회귀로 끝까지 실행돼도 저장소에 결과를 남기지 않게
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)
    with pytest.raises(SystemExit):
        train_cli.evaluate("m", split="tset", splits=tmp_path / "splits", corpus=corpus_path)


def test_evaluate_onnx_fails_when_onnx_older_than_weights(
    tmp_path, corpus, questions, monkeypatch
) -> None:
    """다시 학습한 뒤 ONNX를 다시 변환하지 않았으면 예전 모델로 평가하지 않고 안내한다."""
    monkeypatch.chdir(tmp_path)  # 회귀로 끝까지 실행돼도 저장소에 결과를 남기지 않게
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


def test_compare_command_writes_table(tmp_path, corpus, questions, monkeypatch) -> None:
    """실험 설정의 모델마다 같은 질문으로 평가하고, 모델별 결과와 비교표를 results/에 쓴다."""
    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    exp = tmp_path / "experiments" / "exp_005_base_model_comparison"
    exp.mkdir(parents=True)
    config = exp / "config.yaml"
    config.write_text(
        f"name: base_model_comparison\nquestions: {qpath}\n"
        "models:\n  - intfloat/multilingual-e5-small\n  - google/embeddinggemma-300m\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)

    train_cli.compare(config, corpus=corpus_path, index_dir=tmp_path / "index")

    results = exp / "results"
    table = json.loads((results / "comparison.json").read_text(encoding="utf-8"))
    assert [row["model"] for row in table["models"]] == [
        "intfloat/multilingual-e5-small",
        "google/embeddinggemma-300m",
    ]
    assert table["n"] == 28  # 코퍼스에 없는 질문 1개는 빠진다
    assert set(table["models"][0]["doc"]) >= {"recall@1", "recall@10", "mrr@10"}
    assert (results / "multilingual-e5-small.json").exists()
    markdown = (results / "comparison.md").read_text(encoding="utf-8")
    assert "| google/embeddinggemma-300m |" in markdown
    assert (tmp_path / "index" / "embeddinggemma-300m.sqlite").exists()


def test_compare_command_requires_models(tmp_path, corpus, questions) -> None:
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    config = tmp_path / "config.yaml"
    config.write_text(f"questions: {qpath}\nmodels: []\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_cli.compare(config, corpus=corpus_path, index_dir=tmp_path / "index")


def _fake_expand(queries, cache_path, *, llm=None, min_interval_s=0.0, **_):
    from ragkit.rag.query_expansion import Expansion

    return {
        q: Expansion(
            query=q, expansion="확장어", llm=llm or "fake", input_tokens=10,
            output_tokens=2, latency_s=0.5,
        )
        for q in queries
    }


def test_evaluate_with_expand_searches_expanded_query(tmp_path, corpus, questions, monkeypatch) -> None:
    """--expand면 원문 + 확장어로 검색하고, 결과에 확장 비용과 원문 질문을 남긴다."""
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    embedded: list[str] = []

    def fake_create(model_name, checkpoint_path=None, device=None, backend=None):
        def embed(texts, batch_size=None):
            embedded.extend(texts)
            return np.ones((len(texts), 2)) / np.sqrt(2)

        return embed

    monkeypatch.setattr(train_cli, "create_embedding_fn", fake_create)
    monkeypatch.setattr(train_cli, "expand_queries", _fake_expand)
    out = tmp_path / "result.json"

    train_cli.evaluate(
        "base-model",
        splits=tmp_path / "splits",
        corpus=corpus_path,
        index=tmp_path / "index.sqlite",
        out=out,
        expand=True,
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["expansion"]["llm_calls_per_query"] == 1
    assert result["expansion"]["latency_s_mean"] == 0.5
    first = result["per_question"][0]
    assert "확장어" not in first["query"]  # 원문 질문
    assert first["expanded_query"].endswith("확장어")
    assert any(text.endswith("확장어") for text in embedded)


def test_evaluate_expand_default_out_has_suffix(tmp_path, corpus, questions, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)
    monkeypatch.setattr(train_cli, "expand_queries", _fake_expand)

    train_cli.evaluate(
        "intfloat/multilingual-e5-small", splits=tmp_path / "splits", corpus=corpus_path,
        index=tmp_path / "index.sqlite", expand=True,
    )

    results = tmp_path / "experiments" / "results"
    assert (results / "multilingual-e5-small_test_expand.json").exists()


def test_evaluate_expand_fails_clearly_without_key(tmp_path, corpus, questions, monkeypatch) -> None:
    from ragkit.rag.query_expansion import ExpansionUnavailable

    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    train_cli.split(qpath, corpus=corpus_path, out=tmp_path / "splits")
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)

    def unavailable(*args, **kwargs):
        raise ExpansionUnavailable("키 없음")

    monkeypatch.setattr(train_cli, "expand_queries", unavailable)
    with pytest.raises(SystemExit):
        train_cli.evaluate(
            "base-model", splits=tmp_path / "splits", corpus=corpus_path,
            index=tmp_path / "index.sqlite", out=tmp_path / "r.json", expand=True,
        )


def test_compare_supports_expand_rows(tmp_path, corpus, questions, monkeypatch) -> None:
    """설정의 models 항목에 {model, expand: true}를 쓰면 같은 모델을 쿼리 확장으로도 평가한다."""
    monkeypatch.chdir(tmp_path)
    corpus_path, qpath = _write_inputs(tmp_path, corpus, questions)
    exp = tmp_path / "experiments" / "exp_003_llm_query_expansion"
    exp.mkdir(parents=True)
    config = exp / "config.yaml"
    config.write_text(
        f"name: llm_query_expansion\nquestions: {qpath}\n"
        "models:\n  - intfloat/multilingual-e5-small\n"
        "  - model: intfloat/multilingual-e5-small\n    expand: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_cli, "create_embedding_fn", _fake_create)
    monkeypatch.setattr(train_cli, "expand_queries", _fake_expand)

    train_cli.compare(config, corpus=corpus_path, index_dir=tmp_path / "index")

    results = exp / "results"
    table = json.loads((results / "comparison.json").read_text(encoding="utf-8"))
    assert [row.get("expansion") is not None for row in table["models"]] == [False, True]
    assert (results / "multilingual-e5-small.json").exists()
    assert (results / "multilingual-e5-small_expand.json").exists()
    markdown = (results / "comparison.md").read_text(encoding="utf-8")
    assert "| intfloat/multilingual-e5-small + 쿼리 확장 |" in markdown
    assert "LLM 호출/질문" in markdown


def test_expand_command_explains_quota(tmp_path, corpus, questions, monkeypatch, capsys) -> None:
    """무료 등급 한도(429)면 traceback 대신 받은 만큼은 캐시에 남았고 다시 실행하면 된다고 알린다."""
    from google.genai import errors

    _, qpath = _write_inputs(tmp_path, corpus, questions)

    def quota(*args, **kwargs):
        raise errors.ClientError(429, {"error": {"code": 429, "message": "limit: 20", "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr(train_cli, "expand_queries", quota)
    with pytest.raises(SystemExit):
        train_cli.expand(qpath, cache=tmp_path / "c.jsonl")

    out = capsys.readouterr().out
    assert "한도" in out and "다시 실행" in out and "limit: 20" in out


def test_expand_command_builds_cache(tmp_path, corpus, questions, monkeypatch, capsys) -> None:
    _, qpath = _write_inputs(tmp_path, corpus, questions)
    seen = {}

    def fake(queries, cache_path, *, llm=None, min_interval_s=0.0, **_):
        seen.update(n=len(queries), cache=cache_path, interval=min_interval_s)
        return _fake_expand(queries, cache_path, llm=llm)

    monkeypatch.setattr(train_cli, "expand_queries", fake)

    train_cli.expand(qpath, cache=tmp_path / "c.jsonl", rpm=15)

    assert seen == {"n": 29, "cache": tmp_path / "c.jsonl", "interval": 4.0}
    assert "29개" in capsys.readouterr().out
