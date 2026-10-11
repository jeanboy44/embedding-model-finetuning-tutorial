# ragkit-mcp — 법령 검색 MCP 서버

한국 법령 조문 검색을 LLM 에이전트(Antigravity agy, Claude Code, Claude Desktop 등)의 도구로 제공한다.
공식 `mcp` Python SDK(2.x, `MCPServer`)로 만들었고 stdio로 통신한다.

## 준비

인덱스가 있어야 한다 (저장소 루트에서):

```bash
uv run ragkit index                     # data/processed/index/<모델 키>.sqlite
```

모델·인덱스는 첫 도구 호출 때 연다. 서버 시작은 빠르고, 첫 호출만 몇 초 걸린다.

## 도구

| 도구 | 입력 | 출력 |
|---|---|---|
| `search_laws` | `query, k=5, laws=None` | 가까운 조문 조각: id, title, law_name, article_no, score, text, source_url |
| `get_article` | `doc_id` | 그 조각이 속한 조 전체: title, law_name, source_url, pieces[{id, title, text}] |
| `list_laws` | `theme=None` | 법령 목록: law_name, law_type, theme, doc_count (`laws` 필터에 쓸 정확한 이름) |
| `ask` | `question, laws=None, k=5` | Gemini 답변 + 근거 {answer, sources[{n, id, title}], error}. `GEMINI_API_KEY`가 없으면 error만 채워진다 |

모르는 법령 이름이나 없는 id는 도구 오류로 돌아온다 (비슷한 법령 이름을 안내).

## 다중 사용자 환경 (원격 Streamable HTTP MCP — 권장)

여러 팀원이나 수강생이 함께 쓰는 경우, 서버 1대만 띄워두면 **사용자들은 로컬에 모델·Python·uv를 설치할 필요 없이** URL 하나로 즉시 연결합니다.
(서버에서 모델이 미리 로드되어 콜드 스타트 지연도 전혀 없습니다.)

### 1. 서버 실행 (호스트 1대)

```bash
uv run --package ragkit-mcp ragkit-mcp --transport streamable-http --port 8000
```

### 2. 사용자 클라이언트 등록 (각 사용자 PC)

- **Antigravity CLI (agy)**:
  ```bash
  agy mcp add ragkit-law http://<서버주소>:8000/mcp
  ```

- **Antigravity IDE / Claude Desktop (`mcp_config.json`)**:
  ```json
  {
    "mcpServers": {
      "ragkit-law": {
        "url": "http://<서버주소>:8000/mcp"
      }
    }
  }
  ```

---

## 단일 사용자 로컬 환경 (stdio 프로세스 연결)

각자 컴퓨터에서 직접 프로세스를 띄워 쓰는 방식입니다.

### Antigravity CLI (agy)에 등록

```bash
REPO=/path/to/embedding-model-finetuning-tutorial
agy mcp add --env RAGKIT_PROJECT_ROOT=$REPO ragkit-law $(which uv) run --directory $REPO --package ragkit-mcp ragkit-mcp
```

### Claude Code에 등록

```bash
REPO=/path/to/embedding-model-finetuning-tutorial
claude mcp add ragkit-law -e RAGKIT_PROJECT_ROOT=$REPO -- \
  $(which uv) run --directory $REPO --package ragkit-mcp ragkit-mcp
```

### GUI 앱 (Antigravity IDE · Claude Desktop) 등록

macOS GUI 앱은 터미널의 `PATH`를 상속받지 못하므로 `command`에 반드시 **`uv`의 절대 경로(`which uv`)**를 적어야 합니다.

`~/.gemini/config/mcp_config.json` 또는 `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ragkit-law": {
      "command": "/opt/homebrew/bin/uv",
      "args": [
        "run",
        "--directory", "/path/to/embedding-model-finetuning-tutorial",
        "--package", "ragkit-mcp",
        "ragkit-mcp"
      ],
      "env": {
        "RAGKIT_PROJECT_ROOT": "/path/to/embedding-model-finetuning-tutorial"
      }
    }
  }
}
```

> **콜드 스타트 방지**: `ragkit-mcp`는 첫 호출 시 모델을 지연 로드(Lazy)하고 서버 기동과 동시에 백그라운드에서 모델을 웜업하여 타임아웃을 방지합니다.

## 구조

- `tools.py` — 도구 로직. `Searcher`를 첫 인자로 받는 순수 함수 (테스트는 가짜 Searcher로).
- `server.py` — `MCPServer("ragkit-law")`에 도구를 등록하고 stdio 및 Streamable HTTP를 지원한다.
