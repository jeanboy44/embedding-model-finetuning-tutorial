"""법령 검색 MCP 서버 (stdio): tools.py의 함수를 LLM 도구로 등록한다.

Searcher(모델·인덱스)는 첫 도구 호출 때 한 번 연다. 서버 시작이 빨라 클라이언트가 기다리지 않는다.
"""

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

# RAGKIT_PROJECT_ROOT가 설정되지 않은 경우(GUI 앱 등에서 실행 시) 저장소 루트를 기본값으로 지정
if "RAGKIT_PROJECT_ROOT" not in os.environ:
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / "pyproject.toml").exists():
        os.environ["RAGKIT_PROJECT_ROOT"] = str(repo_root)

import cyclopts
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from ragkit.service import Searcher
from ragkit_mcp import tools

T = TypeVar("T")

INSTRUCTIONS = """\
대한민국 법령(근로기준법, 주택임대차보호법, 최저임금법, 도로교통법 등) 조문을 검색하는 도구입니다. Tools for searching Korean statutes.
[중요 지침]
- 사용자가 한국 법령, 법적 권리, 근로조건(주휴수당, 퇴직금, 연장근로 등), 부동산 임대차(전세 보증금 등)에 대해 질문하면 모델 자체 지식으로만 추측하지 말고 반드시 search_laws 도구를 호출하여 실제 조문을 검색하세요.
- 조문은 항·호 단위 조각(piece)으로 나뉘어 있습니다. search_laws는 조각을 돌려주고, 조 전체는 get_article로 봅니다.
- 특정 법령만 찾으려면 list_laws로 정확한 법령 이름을 확인한 뒤 search_laws의 laws에 넣으세요.
- 답변 시에는 검색된 조문의 제목(예: 근로기준법 제56조)과 조문 id를 반드시 근거로 명시하세요.
- ask는 서버에 LLM이 켜져 있어야 합니다. error가 오면 search_laws 결과로 직접 답하세요.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

server = MCPServer("ragkit-law", instructions=INSTRUCTIONS, log_level="WARNING")

_open_kwargs: dict[str, Any] = {}
_searcher: Searcher | None = None
_lock = threading.Lock()


def get_searcher() -> Searcher:
    """Searcher를 처음 부를 때 열고 이후에는 재사용한다. 열 수 없으면 ToolError."""
    global _searcher
    with _lock:
        if _searcher is None:
            try:
                _searcher = Searcher.open(**_open_kwargs, lazy=True)
            except FileNotFoundError as e:
                raise ToolError(str(e)) from e
            except Exception as e:  # 모델 로드 실패도 이유를 모델에게 보여 준다
                raise ToolError(f"모델·인덱스를 열 수 없습니다: {e}") from e
        return _searcher


def set_searcher(searcher: Searcher | None) -> None:
    """미리 연 Searcher를 쓴다 (테스트용). None이면 다음 호출 때 다시 연다."""
    global _searcher
    _searcher = searcher


def _call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    # 예상한 실패(모르는 법령·없는 id)는 메시지를 그대로 모델에게 보여 준다
    try:
        return fn(get_searcher(), *args, **kwargs)
    except ValueError as e:
        raise ToolError(str(e)) from e


Laws = Annotated[
    list[str] | None,
    Field(description="검색할 법령의 정확한 이름 목록 (list_laws로 확인). 예: ['근로기준법']. 비우면 전체."),
]


@server.tool(annotations=READ_ONLY)
def search_laws(
    query: Annotated[str, Field(description="자연어 질문이나 키워드. 예: '야간 근로 수당', '전세 보증금 우선변제'")],
    k: Annotated[int, Field(description="돌려줄 조각 수", ge=1, le=50)] = 5,
    laws: Laws = None,
) -> list[dict]:
    """질문과 의미가 가까운 한국 법령 조문 조각을 찾는다 (임베딩 검색).
    Semantic search over Korean statute pieces (article paragraphs/items).
    한국 법률이나 일상 권리 문제 질문 시 반드시 이 도구를 호출하여 관련 조문을 검색하세요.

    결과마다 id, title(예: 근로기준법 제56조 ...), law_name, article_no, score, text, source_url이 있다.
    조 전체가 필요하면 id로 get_article을 부른다.
    """
    return _call(tools.search_laws, query, k=k, laws=laws)


@server.tool(annotations=READ_ONLY)
def get_article(
    doc_id: Annotated[str, Field(description="search_laws 결과의 id. 예: '근로기준법_법률_제56조'")],
) -> dict:
    """조각 id가 속한 조(條) 전체를 돌려준다: title, law_name, source_url, pieces[{id, title, text}].
    Returns the whole article (all paragraphs) containing the given piece id.
    """
    return _call(tools.get_article, doc_id)


@server.tool(annotations=READ_ONLY)
def list_laws(
    theme: Annotated[str | None, Field(description="이 테마의 법령만. 예: 'youth'. 비우면 전체.")] = None,
) -> list[dict]:
    """인덱스에 든 법령 목록 (law_name, law_type, theme, doc_count).
    Lists statutes in the index; use law_name values for the `laws` filter of search_laws.
    """
    return _call(tools.list_laws, theme)


@server.tool(annotations=READ_ONLY)
def ask(
    question: Annotated[str, Field(description="법령에 관한 질문")],
    laws: Laws = None,
    k: Annotated[int, Field(description="근거로 쓸 조문 수", ge=1, le=20)] = 5,
) -> dict:
    """검색한 조문만 근거로 LLM이 답한다: {answer, sources[{n, id, title}], error}. 답의 [n]은 sources의 n번째.
    Answers with an LLM grounded on retrieved articles. Needs the LLM enabled on the server;
    if `error` is set, answer yourself from search_laws results instead.
    """
    return _call(tools.ask, question, laws=laws, k=k)


cli = cyclopts.App(name="ragkit-mcp", help="법령 검색 MCP 서버 (stdio 또는 HTTP 원격 다중 사용자).")


@cli.default
def run(
    *,
    transport: Literal["stdio", "streamable-http", "sse"] = "stdio",
    host: str = "0.0.0.0",
    port: int = 8000,
    model: str | None = None,
    checkpoint: Path | None = None,
    backend: str | None = None,
    index: Path | None = None,
) -> None:
    """MCP 서버를 띄운다. stdio 로컬 실행 또는 streamable-http/sse 다중 사용자 원격 실행을 지원한다.

    Args:
        transport: stdio (로컬) | streamable-http (원격 다중 사용자 기본) | sse.
        host: HTTP 서버 바인드 호스트 (기본 0.0.0.0).
        port: HTTP 서버 포트 (기본 8000).
        model: 임베딩 모델 이름. 기본값 Settings.embedding_model_name.
        checkpoint: 파인튜닝한 모델 폴더.
        backend: onnx | torch | st. 기본값 Settings.embedding_backend.
        index: 인덱스 파일. 기본값 data/processed/index/<모델 키>.sqlite.
    """
    _open_kwargs.update(model=model, checkpoint=checkpoint, backend=backend, index_path=index)

    # 백그라운드에서 임베딩 모델 웜업 (콜드 스타트 지연 방지)
    def _warmup_background():
        try:
            get_searcher().warmup()
        except Exception:
            pass

    threading.Thread(target=_warmup_background, daemon=True).start()

    if transport == "stdio":
        server.run("stdio")
    else:
        server.run(transport, host=host, port=port)


def main() -> None:
    """ragkit-mcp 진입점."""
    cli()
