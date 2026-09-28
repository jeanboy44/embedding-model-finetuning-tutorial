# 활용 앱 설계: api · search-cli · mcp · web (2026-09-29)

브랜치: `feat/apps`
관련: `docs/PLAN.md` 2·4단계, `docs/superpowers/specs/2026-09-28-ragkit-restructure-design.md`

## 1. 목표와 결정 사항 (사용자 확정)

ragkit으로 만든 법령 검색 인덱스를 **쓰는** 앱 네 개를 만든다.

| 앱 | 강의 단계 | 역할 |
|---|---|---|
| `apps/api` (ragkit-api) | 2단계 | FastAPI. 검색·답변(SSE 스트리밍)·노트북 저장. web의 유일한 백엔드 |
| `apps/search-cli` (ragkit-search) | 4단계 | 최종 사용자용 검색 CLI. 서버 없이 단독 동작 |
| `apps/mcp` (ragkit-mcp) | 4단계 | MCP 서버. Claude 등이 조문을 검색 |
| `apps/web` | 4단계 | NotebookLM형 React 웹앱. api만 호출 |

- **web 범위 = NotebookLM 방식 A**: 노트북 = 법령 묶음(소스). 노트북 안에서 채팅하면 그 법령에서만 검색해 `[1]` 인용이 달린 답을 스트리밍한다. 인용을 누르면 조문 원문이 소스 패널에 열린다. 답변·메모를 노트로 저장한다. 요약·FAQ 생성(Studio), 사용자 문서 업로드는 범위 밖(이후 확장).
- **저장 = api 서버 SQLite** (`data/app/notebooks.sqlite`). 인증 없음(강의용 로컬 단일 사용자).
- **web 스택 = Vite + React + TypeScript + Tailwind + shadcn/ui + TanStack Query**, 패키지 관리자 pnpm.
- **답변은 SSE 스트리밍**. 검색 결과(인용 후보)를 첫 이벤트로 먼저 보낸다.
- **구조 = 접근 1**: ragkit에 공개 `Searcher`를 두고 api·cli·mcp가 각각 직접 쓴다. cli·mcp는 api 서버 없이 동작한다. web만 api를 쓴다.

## 2. ragkit 추가 (공개 API)

### 2.1 `VectorIndex` (retrieval/index.py)

- `search(where=)`에 **목록 값 = IN** 추가: `{"law_name": ["근로기준법", "최저임금법"]}`. sqlite-vec 0.1.9 vec0 메타데이터 컬럼에서 `IN` 동작을 실제 인덱스로 확인했다. 빈 목록은 조건 없음으로 취급하지 않고 `ValueError`(호출 쪽에서 None으로 넘길 것).
- `get(doc_id) -> SearchHit | None` (score=0.0)
- `get_article(parent_id) -> list[SearchHit]`: 같은 조의 조각을 rowid 순서로. 조각이 아닌 조문은 자신 하나.
- `list_laws() -> list[LawInfo]`: `LawInfo(law_name, law_type, theme, doc_count)`, 테마·이름순.
- 연결은 `check_same_thread=False`로 연다(웹 서버 스레드풀에서 쓰기 위함). 동시 접근은 `Searcher`의 잠금으로 직렬화한다.

### 2.2 Gemini 스트리밍 (models/gemini_client.py)

- `stream_with_usage(prompt, ...) -> Iterator[str | Generation]`: 텍스트 조각(str)을 차례로 내보내고 마지막에 전체 텍스트와 토큰 수를 담은 `Generation` 하나를 내보낸다. 재시도(`_with_retry`)는 스트림을 여는 호출에만 적용한다.

### 2.3 `ragkit.service`

```python
@dataclass
class Searcher:
    index: VectorIndex
    embed_fn: EmbedFn
    profile: ModelProfile

    @classmethod
    def open(cls, model=None, checkpoint=None, backend=None, index_path=None) -> "Searcher"
    def search(self, query, k=5, laws=None) -> list[SearchHit]
    def get(self, doc_id) -> SearchHit | None
    def get_article(self, parent_id) -> list[SearchHit]
    def list_laws(self, theme=None) -> list[LawInfo]
    def resolve_laws(self, names) -> list[str]        # 모르는 이름이면 비슷한 이름을 담아 ValueError
    def answer_stream(self, query, k=5, laws=None, history=None, stream=None) -> Iterator[AnswerEvent]
    def answer(self, ...) -> AnswerResult             # answer_stream을 모은 결과
```

- `open`: `model_key(model, checkpoint)`로 인덱스 경로를 정하고(`ragkit index`와 같은 규칙), 프로필(`get_profile`)의 `format_query`로 쿼리를 만든다. 인덱스가 없으면 `FileNotFoundError`(만드는 명령 안내).
- `laws`가 None 또는 빈 목록이면 전체 코퍼스 검색.
- 이벤트: `HitsEvent(hits)` → `DeltaEvent(text)`… → `DoneEvent(answer, input_tokens, output_tokens, latency_s, error)`. Gemini 키가 없거나 호출이 실패하면 `HitsEvent` 뒤에 `DoneEvent(error=...)`로 끝난다(검색 결과는 보여 줄 수 있게).
- `history`: `[{"role": "user"|"assistant", "content": str}]` 최근 6개까지 프롬프트의 "[이전 대화]" 블록에 넣는다. 검색은 현재 질문만 쓴다. `build_prompt(query, hits, history=None)`로 확장.
- `threading.Lock`으로 index·embed 호출을 직렬화한다.

## 3. apps/api (ragkit_api)

파일: `app.py`(create_app, 라우트), `schemas.py`(pydantic), `store.py`(NotebookStore), `sse.py`(이벤트 직렬화), `cli.py`(진입점 `ragkit-api`, cyclopts + uvicorn).

### 3.1 엔드포인트 (접두어 `/api`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/health` | `{status, model_key, doc_count, llm_available}` |
| GET | `/api/laws?theme=` | 법령 목록 `LawInfo[]` |
| GET | `/api/docs/{doc_id}` | 조문 조각 하나. 없으면 404 |
| GET | `/api/articles/{parent_id}` | 같은 조의 조각 목록. 없으면 404 |
| POST | `/api/search` | `{query, k=5, laws?}` → `{hits}` |
| POST | `/api/answer` | 비스트리밍 답변 `{answer, hits, input_tokens, output_tokens, latency_s, error}` |
| POST | `/api/answer/stream` | SSE |
| GET/POST | `/api/notebooks` | 목록 / 생성 `{title, laws}` |
| GET/PATCH/DELETE | `/api/notebooks/{id}` | 조회 / 수정 `{title?, laws?}` / 삭제(메시지·노트 함께) |
| GET/DELETE | `/api/notebooks/{id}/messages` | 대화 기록 / 비우기 |
| POST | `/api/notebooks/{id}/chat` | `{query, k=5}` → SSE. 노트북의 laws로 검색, 최근 대화를 history로, 사용자·어시스턴트 메시지 저장 |
| GET/POST | `/api/notebooks/{id}/notes` | 노트 목록 / 생성 `{title, content, citations?}` |
| PATCH/DELETE | `/api/notebooks/{id}/notes/{note_id}` | 노트 수정 / 삭제 |

- 검색 결과 형식 `Hit`: `{id, parent_id, title, text, score, law_name, law_type, theme, article_no, article_title, chapter, source_url}`.
- SSE 이벤트: `event: hits` (`{hits}`), `event: delta` (`{text}`), `event: done` (`{answer, input_tokens, output_tokens, latency_s, error, message_id?}`). chat은 `done`에 저장된 어시스턴트 메시지 id를 넣는다.
- 알 수 없는 법령 이름은 422(비슷한 이름 안내). 없는 노트북·노트는 404.
- 시작 시(lifespan) `Searcher.open`으로 모델·인덱스를 한 번 로드한다. 테스트는 `create_app(searcher=..., store=...)`로 가짜를 주입한다.
- CORS: `RAGKIT_API_CORS_ORIGINS`(기본 `http://localhost:5173`). 개발 중 web은 Vite 프록시(`/api` → 8000)를 쓴다.
- `ragkit-api --web-dist apps/web/dist`: 빌드한 web을 같은 서버의 `/`에서 제공(배포 시 한 프로세스). SPA 경로는 index.html로 돌려준다.

### 3.2 저장소 (store.py)

SQLite 파일 하나, 요청마다 연결을 열고 닫는다(스레드 안전).

```
notebooks(id TEXT PK, title, laws JSON, created_at, updated_at)
messages(id TEXT PK, notebook_id FK, role, content, citations JSON, created_at)
notes(id TEXT PK, notebook_id FK, title, content, citations JSON, created_at, updated_at)
```

- id는 uuid4 hex. 시각은 UTC ISO. `ON DELETE CASCADE` + `PRAGMA foreign_keys=ON`.
- citations는 답변 당시 hits(위 `Hit` 형식)를 그대로 저장해, 인덱스를 다시 만들어도 기록이 깨지지 않게 한다.
- 노트북 `updated_at`은 채팅·노트 변경 때도 갱신(목록 정렬용).

## 4. apps/search-cli (ragkit_search)

cyclopts. 진입점 `ragkit-search`.

```
ragkit-search search "야간 근로 수당" [-k 5] [--law 근로기준법 ...] [--json]
ragkit-search ask "질문" [--law ...] [-k 5]      # 스트리밍 출력 후 근거 조문 목록
ragkit-search laws [--theme youth] [--json]
ragkit-search show <doc_id> [--article]          # --article이면 같은 조 전체
```

- 공통 옵션은 Settings(.env / 환경 변수)에서: 모델, 백엔드, `RAGKIT_PROJECT_ROOT`. `--index`로 인덱스 파일을 직접 지정할 수 있다.
- 출력은 표준 라이브러리만으로(ragkit core 외 의존성 없음). `--json`은 기계 처리용.
- 오류(인덱스 없음, 모르는 법령, 키 없음)는 한 줄 안내 + 종료 코드 1.

## 5. apps/mcp (ragkit_mcp)

공식 `mcp` Python SDK의 FastMCP, stdio. 진입점 `ragkit-mcp`.

| 도구 | 입력 | 출력 |
|---|---|---|
| `search_laws` | `query, k=5, laws=None` | 상위 조문(id, title, law_name, score, text) |
| `get_article` | `doc_id` | 같은 조 전체 조각 |
| `list_laws` | `theme=None` | 법령 이름·종류·테마·조각 수 |
| `ask` | `question, laws=None, k=5` | Gemini 답변 + 근거. 키가 없으면 오류 메시지(에이전트는 search_laws로 대신) |

- Searcher는 첫 도구 호출 때 로드한다(서버 시작을 빠르게).
- 도구 로직은 `tools.py`의 순수 함수(`searcher`를 인자로 받음)로 두고 `server.py`는 등록만 한다. 테스트는 순수 함수를 가짜 Searcher로 검증.
- 기존 골격(`embed_tool` 등, `DocumentStore` 기반)은 `ragkit_mcp/legacy.py`로 옮기고 `tutorials/04_product/02_mcp_server.py` import를 고친다.
- README에 Claude Code / Claude Desktop 등록 예시(`uv run --package ragkit-mcp ragkit-mcp`, `RAGKIT_PROJECT_ROOT`).

## 6. apps/web

### 6.1 화면

- `/` 홈: 노트북 카드 목록(제목, 소스 법령 수, 수정 시각) + "새 노트북".
- `/notebooks/:id`: 3단 레이아웃 (좁은 화면에서는 탭 전환)
  - **소스 패널(왼쪽)**: 노트북 법령 목록, "소스 추가" 대화상자(테마 필터·이름 검색·다중 선택), 제거. 인용을 누르면 이 패널이 **조문 보기**로 바뀌어 같은 조 전체를 보여 주고 인용된 조각을 강조한다(법제처 링크 포함). 소스가 없으면 "전체 법령에서 검색" 안내.
  - **채팅(가운데)**: 대화 기록, 입력창, 스트리밍 답변(markdown). 답 속 `[n]`은 클릭 가능한 인용 칩, 답 아래에 근거 조문 칩 목록. 답마다 "노트에 저장". 대화 비우기. 오류(키 없음 등)는 답 자리에 안내 + 검색 결과 칩은 유지.
  - **노트(오른쪽)**: 노트 목록, 새 메모 작성, 편집, 삭제. 답에서 저장한 노트는 인용을 유지해 칩을 누르면 조문 보기로 간다.
- 노트북 제목은 헤더에서 바로 수정.

### 6.2 구조

```
apps/web/
  package.json, vite.config.ts (프록시 /api → http://localhost:8000), tsconfig, components.json
  src/
    api/client.ts        # fetch 래퍼 + 타입 (schemas와 1:1)
    api/sse.ts           # POST SSE 파서 (ReadableStream → 이벤트)
    api/hooks.ts         # TanStack Query 훅
    lib/citations.ts     # 답 텍스트의 [n] → 인용 링크 변환
    components/ui/       # shadcn 컴포넌트
    components/...       # SourcesPanel, ArticleViewer, ChatPanel, Message, NotesPanel, AddSourcesDialog
    pages/Home.tsx, pages/Notebook.tsx
```

- 테스트: vitest로 `sse.ts` 파서(조각 경계에 걸친 이벤트), `citations.ts` 변환. 전체 동작은 실제 api + 인덱스로 Playwright 스모크(수동 검증 단계).

## 7. 기타

- `.gitignore`에 `data/app/`, `apps/web/node_modules/`, `apps/web/dist/`.
- `docs/PLAN.md`: 4단계 스택·web 범위 결정 반영, 미정 사항에서 해당 항목 제거.
- README에 앱 실행 방법(api·web 개발 서버, cli, mcp).
- 범위 밖: 인증, 다중 사용자, Studio 생성물, 문서 업로드, Docker, 레거시 `ragkit` CLI `search/rag` 정리.

## 8. 테스트 전략

- ragkit: 가짜 embed(결정적 벡터)로 만든 임시 인덱스에서 IN 필터, get/get_article/list_laws, resolve_laws, answer_stream 이벤트 순서(가짜 stream 주입), 키 없음 처리.
- api: TestClient + 가짜 Searcher(임시 인덱스 + 가짜 stream) + 임시 store. 노트북 CRUD, chat SSE가 메시지를 저장하는지, 404/422.
- search-cli: cyclopts 앱을 가짜 Searcher로 호출해 출력 확인.
- mcp: tools.py 순수 함수 테스트.
- web: vitest + `pnpm build` 타입 검사.
- 마지막에 실제 인덱스(`multilingual-e5-small.sqlite`)로 api를 띄워 web을 Playwright로 한 바퀴 확인한다.
