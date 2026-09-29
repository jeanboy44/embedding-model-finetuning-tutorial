# ragkit-mcp — 법령 검색 MCP 서버

한국 법령 조문 검색을 LLM 에이전트(Claude Code, Claude Desktop 등)의 도구로 제공한다.
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

## Claude Code에 등록

```bash
REPO=/path/to/embedding-model-finetuning-tutorial
claude mcp add ragkit-law -e RAGKIT_PROJECT_ROOT=$REPO -- \
  uv run --directory $REPO --package ragkit-mcp ragkit-mcp
```

다른 모델의 인덱스를 쓰려면 끝에 옵션을 붙인다 (`ragkit-mcp --help`):

```bash
claude mcp add ragkit-law -e RAGKIT_PROJECT_ROOT=$REPO -- \
  uv run --directory $REPO --package ragkit-mcp ragkit-mcp --model google/embeddinggemma-300m
```

`ask`까지 쓰려면 `-e GEMINI_API_KEY=...`를 더하거나 저장소 `.env`에 넣는다.
키가 없어도 에이전트가 `search_laws` 결과로 직접 답할 수 있다.

## Claude Desktop에 등록

`claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`):

```json
{
  "mcpServers": {
    "ragkit-law": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/embedding-model-finetuning-tutorial",
               "--package", "ragkit-mcp", "ragkit-mcp"],
      "env": { "RAGKIT_PROJECT_ROOT": "/path/to/embedding-model-finetuning-tutorial" }
    }
  }
}
```

Desktop은 PATH가 짧을 수 있으니 `command`에 `uv`의 절대 경로(`which uv`)를 쓰는 편이 안전하다.

## 구조

- `tools.py` — 도구 로직. `Searcher`를 첫 인자로 받는 순수 함수 (테스트는 가짜 Searcher로).
- `server.py` — `MCPServer("ragkit-law")`에 도구를 등록하고 Searcher를 지연 로드한다. `main()`은 stdio.
- `legacy.py` — 강의 04_product/02_mcp_server.py가 쓰는 예전 골격 (`DocumentStore` 기반).
