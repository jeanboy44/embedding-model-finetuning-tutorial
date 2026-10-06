"""
4단계-2: MCP 서버 — Claude 같은 에이전트가 법령을 검색하게 하기 (apps/mcp, ragkit-mcp)
======================================================================================

학습 목표:
- MCP(Model Context Protocol): LLM 에이전트가 외부 도구를 부르는 표준 방식을 이해한다
- 에이전트가 하는 일을 그대로 재현한다: 서버 연결(stdio) → 도구 목록 → 도구 호출
- 도구 설계: 이름·설명·입력 스키마가 곧 에이전트가 읽는 "사용 설명서"다
- Claude Code / Claude Desktop에 등록해 실제로 쓴다

사전 준비:
    uv run ragkit index

실행:
    uv run python lecture/04_product/02_mcp_server.py
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


async def main() -> None:
    # 에이전트(클라이언트)는 서버 프로세스를 띄우고 표준 입출력으로 대화한다
    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(ROOT), "--package", "ragkit-mcp", "ragkit-mcp"],
        env={**os.environ, "RAGKIT_PROJECT_ROOT": str(ROOT)},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        info = await session.initialize()
        section("1. 연결: 서버가 자기소개(instructions)를 보낸다")
        print(f"서버: {info.server_info.name}")
        print((info.instructions or "").strip())

        section("2. 도구 목록: 에이전트는 이 설명을 읽고 언제 무엇을 부를지 정한다")
        for tool in (await session.list_tools()).tools:
            params_ = ", ".join(tool.input_schema.get("properties", {}))
            print(f"- {tool.name}({params_})\n    {(tool.description or '').splitlines()[0]}")

        section("3. 도구 호출: 에이전트가 하는 순서 그대로")
        print('list_laws(theme="youth") → 정확한 법령 이름 확인')
        result = await session.call_tool("list_laws", {"theme": "youth"})
        first = json.loads(result.content[0].text)  # 목록은 항목마다 content 하나로 온다
        print(f"  {len(result.content)}개 법령, 예: {first['law_name']} ({first['law_type']}, 조각 {first['doc_count']}개)")

        print('\nsearch_laws(query="수습 기간 최저임금", laws=["최저임금법"], k=2)')
        found = text_of(await session.call_tool(
            "search_laws", {"query": "수습 기간 최저임금", "laws": ["최저임금법"], "k": 2}))
        print("  " + found[:500].replace("\n", "\n  "))

        print('\nget_article(doc_id="최저임금법_법률_제5조") → 조 전체를 읽고 답한다')
        article = text_of(await session.call_tool("get_article", {"doc_id": "최저임금법_법률_제5조"}))
        print("  " + article[:400].replace("\n", "\n  "))

        print('\n잘못된 입력: search_laws(laws=["없는법"]) → 에이전트가 읽고 고칠 수 있는 오류')
        bad = await session.call_tool("search_laws", {"query": "수당", "laws": ["없는법"]})
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
