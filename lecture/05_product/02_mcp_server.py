"""
실습 5-2 (8교시): MCP 서버 — Claude 같은 에이전트가 법령을 검색하게 하기 (apps/mcp, ragkit-mcp)
================================================================================================

학습 목표:
- MCP(Model Context Protocol): LLM 에이전트가 외부 도구를 부르는 표준 방식을 이해한다
- 에이전트가 하는 일을 그대로 재현한다: 서버 연결(stdio) → 도구 목록 → 도구 호출
- 도구 설계: 이름·설명·입력 스키마가 곧 에이전트가 읽는 "사용 설명서"다
- Claude Code / Claude Desktop에 등록해 실제로 쓴다

서버·네트워크 없이 이 컴퓨터에서 서버 프로세스를 띄워 표준 입출력으로 대화한다 (수십 초).
--run 모드는 없다(받은 인덱스만 읽는다).

사전 준비:
    uv run ragkit index             # 또는 Drive에서 받은 data/processed/index/

실행:
    uv run python lecture/05_product/02_mcp_server.py
"""

import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def text_of(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


def items_of(result) -> list[dict]:
    """도구 결과를 JSON 객체 목록으로. 목록을 돌려주는 도구는 항목마다 content 하나로 온다."""
    return [json.loads(c.text) for c in result.content if getattr(c, "text", "")]


async def main() -> None:
    # 에이전트(클라이언트)는 서버 프로세스를 띄우고 표준 입출력으로 대화한다
    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(ROOT), "--package", "ragkit-mcp", "ragkit-mcp"],
        env={**os.environ, "RAGKIT_PROJECT_ROOT": str(ROOT)},
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        info = await session.initialize()
        section("1. 연결: 서버가 자기소개(instructions)를 보낸다")
        print(f"서버: {info.server_info.name}")
        print((info.instructions or "").strip())

        section("2. 도구 목록: 에이전트는 이 설명을 읽고 언제 무엇을 부를지 정한다")
        for tool in (await session.list_tools()).tools:
            params_ = ", ".join(tool.input_schema.get("properties", {}))
            print(
                f"- {tool.name}({params_})\n    {(tool.description or '').splitlines()[0]}"
            )

        section("3. 도구 호출: 에이전트가 하는 순서 그대로")
        print('list_laws(theme="youth") → 정확한 법령 이름 확인')
        laws = items_of(await session.call_tool("list_laws", {"theme": "youth"}))
        first = laws[0]
        print(
            f"  {len(laws)}개 법령, 예: {first['law_name']} ({first['law_type']}, 조각 {first['doc_count']}개)"
        )

        print('\nsearch_laws(query="수습 기간 최저임금", laws=["최저임금법"], k=2)')
        hits = items_of(
            await session.call_tool(
                "search_laws",
                {"query": "수습 기간 최저임금", "laws": ["최저임금법"], "k": 2},
            )
        )
        print(
            f"  결과마다 필드 {len(hits[0]) if hits else 0}개 (id, title, score, text, source_url …) 중 title · score만:"
        )
        for hit in hits:
            print(f"  {hit['score']:.3f}  {hit['title']}  (id={hit['id']})")

        print('\nget_article(doc_id="최저임금법_법률_제5조") → 조 전체를 읽고 답한다')
        article = items_of(
            await session.call_tool("get_article", {"doc_id": "최저임금법_법률_제5조"})
        )[0]
        print(f"  {article['title']}: 조각 {len(article['pieces'])}개")
        print("  " + article["pieces"][0]["text"][:150].replace("\n", " ") + " …")

        print(
            '\n잘못된 입력: search_laws(laws=["없는법"]) → 에이전트가 읽고 고칠 수 있는 오류'
        )
        bad = await session.call_tool(
            "search_laws", {"query": "수당", "laws": ["없는법"]}
        )
        print(f"  is_error={bad.is_error}: {text_of(bad)[:150]}")


asyncio.run(main())

section("4. 실제 에이전트에 등록")
print(f"""Claude Code:
  claude mcp add ragkit-law -e RAGKIT_PROJECT_ROOT={ROOT} -- \\
    uv run --directory {ROOT} --package ragkit-mcp ragkit-mcp

Claude Desktop (claude_desktop_config.json의 mcpServers):
  "ragkit-law": {{
    "command": "uv",
    "args": ["run", "--directory", "{ROOT}", "--package", "ragkit-mcp", "ragkit-mcp"],
    "env": {{"RAGKIT_PROJECT_ROOT": "{ROOT}"}}
  }}

등록 후 Claude에게: "편의점 알바도 주휴수당 받을 수 있는지 법 조문 근거로 알려줘"
→ Claude가 list_laws / search_laws / get_article을 스스로 골라 부른다.
""")
