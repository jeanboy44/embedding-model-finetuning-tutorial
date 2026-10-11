# 법령 검색 화면 — HTML 한 장 (apps/web-lite)

빌드 없는 HTML 파일 하나입니다. 검색 API 서버(`apps/api`, ragkit-api)가 같은 주소에서 이 파일을 함께 주고, 화면은 `/api/health` · `/api/laws` · `/api/search` · `/api/articles/{조 id}`만 부릅니다. 모델과 인덱스는 서버에만 있습니다.

디자인은 3교시의 오프라인 검색 파일(`apps/offline-search/src/template.html`)과 같습니다(shadcn/ui 흑백 톤). 다른 점은 무거운 것이 어디 있느냐입니다.

| | 오프라인 파일 (`apps/offline-search`) | 이 화면 (`apps/web-lite`) |
|---|---|---|
| 모델 · 조문 벡터 | 파일 안 (56MB) | 서버 |
| 질문 임베딩 | 브라우저 (onnxruntime-web) | 서버 |
| 필요한 것 | 파일 하나 | 서버 한 프로세스 + 이 파일 |
| 모델 바꾸기 | 파일을 다시 만든다 | 서버만 다시 띄운다 |

## 실행

저장소 루트에서. 인덱스(`uv run ragkit index` 또는 Drive에서 받은 것)가 있어야 합니다.

```bash
uv run --package ragkit-api ragkit-api --web-dist apps/web-lite                 # http://127.0.0.1:8000
uv run --package ragkit-api ragkit-api --web-dist apps/web-lite --checkpoint models/finetuned/002_hard_negatives --backend torch   # 파인튜닝 모델로
```

8교시 실습 스크립트 `lecture/08_product/04_web_app.py`가 이 명령으로 서버를 띄우고 응답을 확인합니다.

## 화면

- 질문 → `POST /api/search {query, k, laws}` → 조문 조각(항 · 호) 순위. 카드마다 순위, 제목, 코사인 유사도, 테마 · 법령 종류 · 조각 id.
- 테마 · 법령 종류 · 법령 필터는 `/api/laws` 목록으로 채우고, 고른 법령 이름들을 `laws`로 보냅니다.
- "전문 보기"는 받은 본문을 펼치고, "같은 조 전체"는 `/api/articles/{parent_id}`로 그 조의 조각을 모두 가져옵니다.
- 답변 생성(LLM)은 강의에서 쓰지 않습니다. 검색 결과까지만 보여 줍니다.

NotebookLM처럼 노트북 · 채팅 · 노트까지 있는 React 화면은 `apps/web`입니다(선택 실습).
