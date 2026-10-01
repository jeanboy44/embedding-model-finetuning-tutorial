"""Searcher 트레이싱: 검색 한 번·답변 한 번이 트레이스 하나로 남고, 대화는 세션으로 묶인다."""

from pathlib import Path

import pytest

from ragkit.config import get_settings

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def traces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """임시 SQLite 저장소로 MLflow를 켜고, 기록된 트레이스를 돌려주는 함수를 준다."""
    monkeypatch.chdir(tmp_path)  # 아티팩트 기본 위치(cwd/mlruns)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_TRACE_EXPERIMENT", "svc")
    get_settings.cache_clear()

    def fetch():
        mlflow.flush_trace_async_logging()
        exp = mlflow.get_experiment_by_name("svc")
        found = mlflow.search_traces(locations=[exp.experiment_id], return_type="list")
        return [mlflow.get_trace(t.info.trace_id) for t in found]

    yield fetch
    get_settings.cache_clear()


def _spans(trace) -> dict[str, object]:
    return {span.name: span for span in trace.data.spans}


def test_search_is_one_trace_with_retrieved_documents(searcher, traces) -> None:
    """검색 한 번 = 트레이스 하나: 질문 임베딩(EMBEDDING) + 조문 검색(RETRIEVER, 찾은 조문 본문 포함)."""
    searcher.search("휴일 휴일", k=2, source="search-cli")

    [trace] = traces()
    spans = _spans(trace)
    assert spans["search"].span_type == "RETRIEVER"
    assert spans["embed_query"].span_type == "EMBEDDING"
    docs = spans["search"].outputs
    assert docs[0]["id"] == "제55조" and docs[0]["page_content"] == "휴일 휴일 휴일"
    assert trace.info.tags["source"] == "search-cli"


def test_answer_trace_links_search_and_llm_with_session(searcher, fake_stream, traces) -> None:
    """답변 한 번 = 트레이스 하나: answer(CHAIN) 아래 search(RETRIEVER)와 llm(CHAT_MODEL, 토큰 수)."""
    list(searcher.answer_stream("휴일 수당", k=2, session_id="nb-1", user="kim", source="api"))

    [trace] = traces()
    spans = _spans(trace)
    root = spans["answer"]
    assert root.parent_id is None
    assert spans["search"].parent_id == root.span_id
    assert spans["llm"].parent_id == root.span_id
    assert spans["llm"].span_type == "CHAT_MODEL"
    assert spans["llm"].attributes["mlflow.chat.tokenUsage"]["input_tokens"] == 20
    assert spans["llm"].inputs["messages"][0]["content"] == fake_stream.prompts[0]
    assert root.outputs["answer"] == "휴일에는 가산 임금을 받습니다 [1]"
    assert trace.info.request_metadata["mlflow.trace.session"] == "nb-1"
    assert trace.info.request_metadata["mlflow.trace.user"] == "kim"


def test_notebook_chat_turns_share_one_session(searcher, traces, tmp_path: Path) -> None:
    """노트북 채팅: 질문마다 트레이스 하나, 같은 노트북이면 같은 세션."""
    from fastapi.testclient import TestClient
    from ragkit_api import create_app
    from ragkit_api.store import NotebookStore

    client = TestClient(create_app(searcher=searcher, store=NotebookStore(tmp_path / "nb.sqlite")))
    nb = client.post("/api/notebooks", json={"title": "휴일", "laws": ["근로기준법"]}).json()
    for query in ("휴일 수당", "더 알려줘"):
        with client.stream("POST", f"/api/notebooks/{nb['id']}/chat", json={"query": query}) as res:
            res.read()

    found = traces()
    assert len(found) == 2
    assert {t.info.request_metadata["mlflow.trace.session"] for t in found} == {nb["id"]}
    assert {t.info.tags["source"] for t in found} == {"api"}


def test_tracing_off_leaves_no_trace(searcher, monkeypatch: pytest.MonkeyPatch) -> None:
    """추적 주소가 없으면 검색·답변이 그대로 동작한다."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    get_settings.cache_clear()

    assert searcher.search("휴일", k=1)[0].id == "제55조"
    assert searcher.answer("휴일").answer
