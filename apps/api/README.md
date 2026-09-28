# ragkit-api (apps/api)

법령 검색·근거 답변·노트북을 제공하는 FastAPI 서버입니다. 강의 2단계 산출물이고, `apps/web`의 유일한 백엔드입니다.

```bash
uv run --package ragkit-api ragkit-api                        # http://127.0.0.1:8000, 문서 /docs
uv run --package ragkit-api ragkit-api --checkpoint experiments/exp_002_finetuned/model   # 파인튜닝 모델
uv run --package ragkit-api ragkit-api --web-dist apps/web/dist                          # 빌드한 web도 함께
```

- 시작할 때 `Searcher.open()`으로 모델과 인덱스(`data/processed/index/<모델>.sqlite`)를 한 번 로드합니다. 인덱스가 없으면 `uv run ragkit index` 안내와 함께 멈춥니다.
- 답변에는 `.env`의 `GEMINI_API_KEY`가 필요합니다. 없으면 `/api/health`의 `llm_available`이 false이고, 답변 요청은 검색 결과와 `error`만 돌려줍니다.
- 노트북·대화·노트는 `data/app/notebooks.sqlite`(`--db`로 변경)에 저장합니다. 인증은 없습니다(로컬 단일 사용자).

## 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/health` | 모델 키, 문서 수, LLM 사용 가능 여부 |
| GET | `/api/laws?theme=` | 법령 목록 |
| GET | `/api/docs/{doc_id}` · `/api/articles/{parent_id}` | 조문 조각 / 같은 조 전체 |
| POST | `/api/search` | `{query, k, laws?}` → `{hits}` |
| POST | `/api/answer` · `/api/answer/stream` | 근거 답변 (JSON / SSE) |
| CRUD | `/api/notebooks[/{id}]` | 노트북 `{title, laws}` |
| GET·DELETE | `/api/notebooks/{id}/messages` | 대화 기록 |
| POST | `/api/notebooks/{id}/chat` | 노트북 법령 안에서 답변(SSE), 대화 저장 |
| CRUD | `/api/notebooks/{id}/notes[/{note_id}]` | 노트 |

SSE 이벤트 순서: `hits`(근거 조문) → `delta`(답 조각)… → `done`(전체 답, 토큰 수, 지연, error).

```bash
curl -N -X POST localhost:8000/api/answer/stream -H 'content-type: application/json' \
  -d '{"query": "근로계약서 안 쓰면 처벌받아?", "laws": ["근로기준법"]}'
```
