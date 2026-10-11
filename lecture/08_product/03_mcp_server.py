"""
실습 4-3 (8교시): MCP 서버 — 에이전트를 붙이는 두 번째 방법 (apps/mcp, ragkit-mcp)
================================================================================================

학습 목표:
- MCP(Model Context Protocol): LLM 에이전트가 외부 도구를 부르는 표준 방식을 이해한다
- 에이전트가 하는 일을 그대로 재현한다: 서버 연결(stdio) → 도구 목록 → 도구 호출
- 도구 설계: 이름·설명·입력 스키마가 곧 에이전트가 읽는 "사용 설명서"다
- Antigravity CLI(agy) / Claude Code / Claude Desktop에 등록해 실제로 쓴다

서버·네트워크 없이 이 컴퓨터에서 서버 프로세스를 띄워 표준 입출력으로 대화한다 (수십 초).
--run 모드는 없다(받은 인덱스만 읽는다).

무엇을 하나 (이 스크립트가 에이전트 역할, apps/mcp의 ragkit-mcp가 서버 역할이다):
    1. 연결: `uv run --package ragkit-mcp ragkit-mcp`를 자식 프로세스로 띄우고 initialize → 서버 이름과 instructions(사용 지침)
    2. 도구 목록: list_tools → search_laws · get_article · list_laws · ask의 이름 · 입력값 · 설명 첫 줄
       (ask는 LLM 답변 도구라 이 강의에서 쓰지 않는다)
    3. 도구 호출: list_laws(theme) → search_laws(query, laws, k) → get_article(doc_id), 그리고 잘못된 법령 이름을 준 오류 응답
    4. 등록 명령 출력: 원격(streamable-http) · 로컬(stdio) 두 방법을 이 컴퓨터 경로로. CLI + 스킬(실습 4-2)과 비교

사전 준비:
    uv run ragkit index             # 또는 Drive에서 받은 data/processed/index/

실행:
    uv run python lecture/08_product/03_mcp_server.py
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

ROOT = Path(__file__).resolve().parents[2]
UV_BIN = shutil.which("uv") or "uv"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def text_of(result: CallToolResult) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


def items_of(result: CallToolResult) -> list[dict]:
    """도구 결과를 JSON 객체 목록으로. 목록을 돌려주는 도구는 항목마다 content 하나로 온다."""
    return [json.loads(c.text) for c in result.content if getattr(c, "text", "")]


async def main() -> None:
    # 에이전트(클라이언트)는 서버 프로세스를 띄우고 표준 입출력으로 대화한다.
    # 아래 등록 명령(4번)에 적는 것과 같은 명령 · 환경 변수다 (RAGKIT_PROJECT_ROOT로 인덱스 위치를 찾는다)
    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(ROOT), "--package", "ragkit-mcp", "ragkit-mcp"],
        env={**os.environ, "RAGKIT_PROJECT_ROOT": str(ROOT)},
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        # ------------------------------------------------------------
        # 1. 연결 (initialize): 서버가 이름과 instructions를 보낸다
        # ------------------------------------------------------------
        # instructions는 apps/mcp 서버에 적힌 사용 지침이다. 에이전트는 이 글을 읽고
        # "법령 질문이면 추측하지 말고 search_laws를 부른다"는 규칙을 안다.
        # 출력에서 볼 것: 서버 이름 ragkit-law, 지침 다섯 줄
        info = await session.initialize()
        section("1. 연결: 서버가 자기소개(instructions)를 보낸다")
        print(f"서버: {info.server_info.name}")
        print((info.instructions or "").strip())

        # ------------------------------------------------------------
        # 2. 도구 목록 (list_tools): 이름 · 입력 스키마 · 설명
        # ------------------------------------------------------------
        # 에이전트가 받는 "사용 설명서"는 이것이 전부다. 도구 이름과 설명이 곧 설계다.
        # 출력에서 볼 것: 도구 4개. search_laws(query, k, laws) · get_article(doc_id) · list_laws(theme) · ask(…)
        # ask는 서버에 LLM이 켜져 있어야 하는 답변 도구라 이 강의에서 쓰지 않는다
        section("2. 도구 목록: 에이전트는 이 설명을 읽고 언제 무엇을 부를지 정한다")
        for tool in (await session.list_tools()).tools:
            # 입력 스키마의 인자 이름만
            params_ = ", ".join(tool.input_schema.get("properties", {}))
            print(
                f"- {tool.name}({params_})\n    {(tool.description or '').splitlines()[0]}"
            )

        # ------------------------------------------------------------
        # 3. 도구 호출 (call_tool): 실습 4-1의 laws → search → show와 같은 순서
        # ------------------------------------------------------------
        # 결과는 content 목록으로 오고, 항목마다 JSON 한 건이다(items_of가 파싱). 검색은 실습 4-1과
        # 같은 Searcher라 점수도 같다. 마지막은 틀린 법령 이름 "근로기준"을 일부러 줘 오류 응답을 본다.
        # 출력에서 볼 것: youth 법령 32개, 최저임금법 제5조(0.898) · 제31조, 제5조 조각 1개, is_error=True와 비슷한 이름 안내
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
            '\n잘못된 입력: search_laws(laws=["근로기준"]) → 비슷한 이름을 알려 줘 에이전트가 고쳐 다시 부른다'
        )
        bad = await session.call_tool(
            "search_laws", {"query": "수당", "laws": ["근로기준"]}
        )
        # 예외가 아니라 오류 결과로 돌아온다 (is_error=True)
        print(f"  is_error={bad.is_error}: {text_of(bad)[:150]}")


asyncio.run(main())

# ============================================================
# 4. 실제 에이전트에 등록: 명령을 이 컴퓨터의 경로로 출력한다
# ============================================================
# 방법 A는 서버 한 대를 streamable-http로 띄우고 여러 사용자가 URL로 붙는다.
# 방법 B는 각자의 컴퓨터에서 위 1번과 같은 stdio 프로세스를 등록한다 (ROOT · UV_BIN이 실제 경로로 들어간다).
# 출력에서 볼 것: 복사해 쓸 등록 명령, 등록 뒤 던질 질문, CLI + 스킬과의 비교표
section("4. 실제 에이전트에 등록")
print(f"""[방법 A] 여러 사용자가 함께 쓰는 경우 (원격 Streamable HTTP — 권장):
  1. 서버 1대 기동:
     uv run --package ragkit-mcp ragkit-mcp --transport streamable-http --port 8000

  2. 사용자 PC에서 등록 (로컬 설치 불필요):
     Antigravity CLI (agy):
       agy mcp add ragkit-law http://<서버주소>:8000/mcp
     Claude Desktop / Antigravity IDE (mcp_config.json):
       "ragkit-law": {{"url": "http://<서버주소>:8000/mcp"}}

[방법 B] 내 컴퓨터에서 직접 프로세스를 띄우는 경우 (로컬 stdio):
  Antigravity CLI (agy):
    agy mcp add --env RAGKIT_PROJECT_ROOT={ROOT} ragkit-law $(which uv) run --directory {ROOT} --package ragkit-mcp ragkit-mcp

Claude Code:
  claude mcp add ragkit-law -e RAGKIT_PROJECT_ROOT={ROOT} -- \\
    $(which uv) run --directory {ROOT} --package ragkit-mcp ragkit-mcp

Claude Desktop / Antigravity IDE (mcp_config.json):
  "ragkit-law": {{
    "command": "{UV_BIN}",
    "args": ["run", "--directory", "{ROOT}", "--package", "ragkit-mcp", "ragkit-mcp"],
    "env": {{"RAGKIT_PROJECT_ROOT": "{ROOT}"}}
  }}

등록 후 agy 또는 claude에게: "편의점 알바도 주휴수당 받을 수 있는지 법 조문 근거로 알려줘"
→ 에이전트가 list_laws / search_laws / get_article을 스스로 골라 부른다.

CLI + 스킬(실습 4-2)과 비교:
  MCP          서버 프로세스를 등록한다. 도구 이름·설명·입력 스키마로 에이전트가 쓰는 법을 안다.
               MCP를 지원하는 앱(Antigravity, Claude Desktop 등)이면 어디서나, 터미널이 없어도 쓴다
  CLI + 스킬   SKILL.md 한 파일을 둔다. 에이전트가 셸로 CLI를 부르고, 사람도 같은 명령을 쓴다.
               셸을 쓰는 코딩 에이전트(Antigravity agy, Gemini CLI, Claude Code)에 가볍게 붙는다
같은 Searcher를 쓰므로 검색 결과는 같다. 차이는 연결 방식과 쓰는 곳이다.
""")
