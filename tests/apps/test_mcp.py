"""apps/mcp 테스트: 도구 순수 함수(가짜 Searcher) + MCP 서버 도구 등록·호출."""

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from ragkit_mcp import server as mcp_server
from ragkit_mcp import tools


def test_search_laws_returns_pieces(searcher) -> None:
    hits = tools.search_laws(searcher, "휴일", k=2)
    assert len(hits) == 2
    assert set(hits[0]) == {"id", "title", "law_name", "article_no", "score", "text", "source_url"}
    assert hits[0]["id"] == "제55조" and hits[0]["law_name"] == "근로기준법"


def test_search_laws_filter_and_unknown_law(searcher) -> None:
    assert [h["id"] for h in tools.search_laws(searcher, "임금", laws=["최저임금법"])] == ["제6조"]
    with pytest.raises(ValueError, match="모르는 법령: 임금법"):
        tools.search_laws(searcher, "임금", laws=["임금법"])


def test_get_article_collects_all_pieces(searcher) -> None:
    article = tools.get_article(searcher, "제56조_제2항")
    assert article["law_name"] == "근로기준법"
    assert [p["id"] for p in article["pieces"]] == ["제56조_제1항", "제56조_제2항"]
    with pytest.raises(ValueError, match="없는 조문 id"):
        tools.get_article(searcher, "제999조")


def test_list_laws_by_theme(searcher) -> None:
    assert tools.list_laws(searcher, "tax") == [
        {"law_name": "최저임금법", "law_type": "법률", "theme": "tax", "doc_count": 1}
    ]
    assert len(tools.list_laws(searcher)) == 3


def test_ask_with_and_without_llm(searcher, searcher_without_llm) -> None:
    result = tools.ask(searcher, "휴일 근로", k=2)
    assert result["answer"] == "휴일에는 가산 임금을 받습니다 [1]" and result["error"] is None
    assert [s["n"] for s in result["sources"]] == [1, 2]

    result = tools.ask(searcher_without_llm, "휴일")
    assert result["answer"] == "" and "GEMINI_API_KEY" in result["error"]
    assert result["sources"][0] == {"n": 1, "id": "제55조", "title": "근로기준법 제55조"}


@pytest.fixture
def served(searcher):
    mcp_server.set_searcher(searcher)
    yield mcp_server.server
    mcp_server.set_searcher(None)


def test_server_lists_four_tools(served) -> None:
    listed = asyncio.run(served.list_tools())
    assert {t.name for t in listed} == {"search_laws", "get_article", "list_laws", "ask"}
    search = next(t for t in listed if t.name == "search_laws")
    assert "list_laws" in search.input_schema["properties"]["laws"]["description"]


def test_server_calls_tool_and_surfaces_errors(served) -> None:
    result = asyncio.run(served.call_tool("list_laws", {"theme": "tax"}))
    assert not result.is_error
    assert "최저임금법" in json.dumps(result.structured_content, ensure_ascii=False)

    with pytest.raises(ToolError, match="모르는 법령: 임금법"):
        asyncio.run(served.call_tool("search_laws", {"query": "임금", "laws": ["임금법"]}))
