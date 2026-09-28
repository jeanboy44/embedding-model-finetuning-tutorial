"""apps/api 테스트: 검색·조회·답변 SSE·노트북 CRUD (가짜 Searcher, 임시 저장소)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ragkit_api import create_app
from ragkit_api.store import NotebookStore


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def client(searcher, tmp_path: Path) -> TestClient:
    app = create_app(searcher=searcher, store=NotebookStore(tmp_path / "nb.sqlite"))
    with TestClient(app) as c:
        yield c


def test_health_and_laws(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health == {"status": "ok", "model_key": "fake", "doc_count": 5, "llm_available": True}

    laws = client.get("/api/laws").json()
    assert [law["law_name"] for law in laws] == ["최저임금법", "근로기준법", "주택임대차보호법"]
    assert client.get("/api/laws", params={"theme": "tax"}).json()[0]["doc_count"] == 1


def test_search_with_law_filter(client: TestClient) -> None:
    res = client.post("/api/search", json={"query": "임금", "k": 5, "laws": ["근로기준법"]})

    assert res.status_code == 200
    hits = res.json()["hits"]
    assert {h["law_name"] for h in hits} == {"근로기준법"}
    assert hits[0]["source_url"].startswith("https://www.law.go.kr")


def test_unknown_law_is_422_with_suggestion(client: TestClient) -> None:
    res = client.post("/api/search", json={"query": "임금", "laws": ["근로기준"]})

    assert res.status_code == 422
    assert "근로기준법" in res.json()["detail"]


def test_doc_and_article(client: TestClient) -> None:
    assert client.get("/api/docs/제4조").json()["law_name"] == "주택임대차보호법"
    assert client.get("/api/docs/없음").status_code == 404
    article = client.get("/api/articles/제56조").json()
    assert [h["id"] for h in article] == ["제56조_제1항", "제56조_제2항"]
    assert client.get("/api/articles/없음").status_code == 404


def test_answer_stream_sends_hits_deltas_done(client: TestClient) -> None:
    with client.stream("POST", "/api/answer/stream", json={"query": "휴일", "k": 2}) as res:
        assert res.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(res.read().decode())

    assert [name for name, _ in events] == ["hits", "delta", "delta", "done"]
    assert len(events[0][1]["hits"]) == 2
    assert events[-1][1]["answer"] == "휴일에는 가산 임금을 받습니다 [1]"
    assert events[-1][1]["error"] is None


def test_answer_non_streaming(client: TestClient) -> None:
    body = client.post("/api/answer", json={"query": "휴일", "k": 1}).json()

    assert body["answer"].endswith("[1]") and len(body["hits"]) == 1 and body["output_tokens"] == 5


def test_answer_without_llm_reports_error(searcher_without_llm, tmp_path: Path) -> None:
    app = create_app(searcher=searcher_without_llm, store=NotebookStore(tmp_path / "nb.sqlite"))
    with TestClient(app) as c:
        body = c.post("/api/answer", json={"query": "휴일"}).json()

    assert body["answer"] == "" and "GEMINI_API_KEY" in body["error"] and body["hits"]


def test_notebook_crud(client: TestClient) -> None:
    created = client.post("/api/notebooks", json={"title": "근로", "laws": ["근로기준법"]})
    assert created.status_code == 201
    nb = created.json()

    assert client.get("/api/notebooks").json()[0]["id"] == nb["id"]
    updated = client.patch(f"/api/notebooks/{nb['id']}", json={"laws": ["근로기준법", "최저임금법"]}).json()
    assert updated["laws"] == ["근로기준법", "최저임금법"] and updated["title"] == "근로"
    assert client.patch(f"/api/notebooks/{nb['id']}", json={"laws": ["없는법"]}).status_code == 422
    assert client.delete(f"/api/notebooks/{nb['id']}").status_code == 204
    assert client.get(f"/api/notebooks/{nb['id']}").status_code == 404
    assert client.delete(f"/api/notebooks/{nb['id']}").status_code == 404


def test_chat_searches_notebook_laws_and_saves_messages(client: TestClient, fake_stream) -> None:
    nb = client.post("/api/notebooks", json={"title": "t", "laws": ["최저임금법"]}).json()

    with client.stream("POST", f"/api/notebooks/{nb['id']}/chat", json={"query": "임금", "k": 3}) as res:
        events = parse_sse(res.read().decode())

    hits = events[0][1]["hits"]
    assert {h["law_name"] for h in hits} == {"최저임금법"}
    done = events[-1][1]
    messages = client.get(f"/api/notebooks/{nb['id']}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["id"] == done["message_id"]
    assert messages[1]["citations"][0]["id"] == hits[0]["id"]

    # 두 번째 질문은 이전 대화를 프롬프트에 넣는다
    with client.stream("POST", f"/api/notebooks/{nb['id']}/chat", json={"query": "더 알려줘"}) as res:
        res.read()
    assert "[이전 대화]\n사용자: 임금" in fake_stream.prompts[-1]

    assert client.delete(f"/api/notebooks/{nb['id']}/messages").status_code == 204
    assert client.get(f"/api/notebooks/{nb['id']}/messages").json() == []


def test_chat_on_missing_notebook_is_404(client: TestClient) -> None:
    assert client.post("/api/notebooks/none/chat", json={"query": "q"}).status_code == 404


def test_notes_crud(client: TestClient) -> None:
    nb = client.post("/api/notebooks", json={"title": "t"}).json()
    base = f"/api/notebooks/{nb['id']}/notes"
    citation = client.get("/api/docs/제55조").json()

    note = client.post(base, json={"title": "휴일", "content": "주 1회", "citations": [citation]}).json()
    assert note["citations"][0]["id"] == "제55조"

    edited = client.patch(f"{base}/{note['id']}", json={"content": "주 1회 이상"}).json()
    assert edited["content"] == "주 1회 이상" and edited["title"] == "휴일"
    assert client.get(base).json()[0]["id"] == note["id"]
    assert client.patch(f"{base}/none", json={"title": "x"}).status_code == 404
    assert client.delete(f"{base}/{note['id']}").status_code == 204
    assert client.get(base).json() == []


def test_serves_built_web_with_spa_fallback(searcher, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    app = create_app(searcher=searcher, store=NotebookStore(tmp_path / "nb.sqlite"), web_dist=dist)

    with TestClient(app) as c:
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/notebooks/abc").text == "<html>app</html>"
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/api/nothing").status_code == 404
