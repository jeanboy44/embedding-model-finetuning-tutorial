# 법령 노트 (apps/web)

NotebookLM처럼 쓰는 법령 검색 웹앱입니다. 강의 4단계 산출물로, 백엔드는 `apps/api`(ragkit-api) 하나만 씁니다.

- **노트북 = 법령 묶음**: 노트북마다 소스 법령을 고르면, 질문은 그 법령 안에서만 찾아 답합니다. 고르지 않으면 전체 법령에서 찾습니다.
- **근거 있는 답**: 답은 SSE로 흘러나오고 `[1]` 같은 인용 번호가 붙습니다. 번호를 누르면 왼쪽 소스 패널에 조문 원문이 열리고, 항·호로 나뉜 조문이면 인용한 부분이 강조됩니다.
- **노트**: 좋은 답은 "노트에 저장"으로 모읍니다. 인용도 함께 남습니다. 직접 메모도 쓸 수 있습니다.
- 노트북·대화·노트는 api 서버의 SQLite(`data/app/notebooks.sqlite`)에 저장됩니다.

스택: Vite, React 19, TypeScript, Tailwind CSS v4, shadcn/ui(radix), TanStack Query, react-markdown.

## 실행

먼저 인덱스(`uv run ragkit index`)가 있어야 하고, 답변 생성에는 `.env`의 `GEMINI_API_KEY`가 필요합니다. 키가 없으면 검색 결과(근거 조문)만 보여 줍니다.

```bash
# 터미널 1: API (저장소 루트에서)
uv run --package ragkit-api ragkit-api            # http://127.0.0.1:8000, 문서 /docs

# 터미널 2: 개발 서버 (/api 요청은 8000으로 프록시)
cd apps/web
pnpm install
pnpm dev                                          # http://localhost:5173
```

API를 다른 주소에서 띄웠다면 `RAGKIT_API_URL=http://host:port pnpm dev`.

### 배포: 한 프로세스로

```bash
cd apps/web && pnpm build && cd ../..
uv run --package ragkit-api ragkit-api --web-dist apps/web/dist   # 화면과 API를 같은 포트에서
```

## 개발

```bash
pnpm test    # SSE 파서, 인용 변환 (vitest)
pnpm build   # 타입 검사 + 빌드
pnpm lint    # oxlint
```

```
src/
  api/types.ts      # apps/api schemas.py와 1:1
  api/client.ts     # fetch 래퍼, 채팅 SSE
  api/sse.ts        # POST 응답의 SSE 파서
  api/hooks.ts      # TanStack Query 훅, useChat(스트리밍 한 턴)
  lib/citations.ts  # 답 속 [n] → 인용 칩
  components/       # SourcesPanel · ArticleViewer · ChatPanel · NotesPanel · AddSourcesDialog
  pages/            # Home(노트북 목록), Notebook(3단 화면, 좁으면 탭)
```
