# 실습 자료 재구성 계획 (tutorials → handson)

상태: **계획** (2026-09-30). 아직 옮기지 않았다. 교시 구성은 `docs/PLAN.md` "교시 구성"을 따른다.

## 폴더 규칙

```
handson/
  00_setup/      # 실습 0 (3교시)
  01_data/       # 실습 1 (4교시)
  02_evaluate/   # 실습 2 (5교시)
  03_train/      # 실습 3 (6교시)
  04_optimize/   # 실습 4 (7교시)
  05_product/    # 실습 5 (8교시)
  appendix/      # 선택 실습 (교시 밖)
```

- 폴더마다 `SLIDE.md` 1개 + 번호 붙은 실행 스크립트 `01_*.py`, `02_*.py` …
- `SLIDE.md`: 그 교시에 하는 업무를 설명하는 장표 내용. 맨 위에 **2022년의 이 업무 / 2026년의 이 업무**를 한 줄씩, 이어서 목표·흐름·실행할 스크립트·확인할 결과
- 스크립트는 두 모드를 둔다: 기본은 **"받은 산출물로 확인"**(Drive로 배포한 모델·인덱스·질문·split 사용), `--run`이면 **"작은 부분집합으로 직접 실행"**
- 스크립트 머리 docstring 형식은 지금 튜토리얼과 같다(학습 목표 · 사전 준비 · 실행)
- 실행은 저장소 루트에서 `uv run python handson/<폴더>/<파일>.py`

## 교시별 내용과 기존 파일 매핑

| 폴더 | 교시 | 업무 | 스크립트 (← 기존 파일) | 담당 |
|---|---|---|---|---|
| `00_setup` | 3 | 환경 설정, 완성품 먼저 써 보기, RAG 필요성 | `01_check_env.py` (신규: 설치·모델·인덱스·API 키 점검)<br>`02_try_product.py` (신규: 웹·CLI·MCP를 base vs 파인튜닝으로 나란히)<br>`03_why_rag.py` ← `tutorials/01_ds_core/01_build_rag.py` | f7 |
| `01_data` | 4 | 코퍼스, 질문 생성(Claude 스킬), 법령 단위 분할 | (신규) | 42 |
| `02_evaluate` | 5 | 평가셋·지표, 베이스 모델 비교(실험 005) | (신규) | 42 |
| `03_train` | 6 | 학습 설명 + 맛보기, 받은 모델로 평가, 쿼리 확장 비교표 | (신규, `feat/finetune-runs` 결과 사용) | 42 |
| `04_optimize` | 7 | API 짧은 복습 → ONNX · INT8 · 어휘 가지치기 · 비교표 | `01_api_review.py` ← `tutorials/02_api/01_search_api.py` + `02_streaming_and_notebooks.py` (한 파일로 압축)<br>`02_onnx_quantize_prune.py` ← `tutorials/03_optimize/02_onnx_and_quantize.py`<br>`03_bench_and_serve.py` ← `tutorials/03_optimize/03_bench_and_deploy.py` | f7 |
| `05_product` | 8 | CLI · MCP · 웹, 운영 모니터링(MLflow), 마무리 | `01_search_cli.py` ← `tutorials/04_product/01_search_cli.py`<br>`02_mcp_server.py` ← `tutorials/04_product/02_mcp_server.py`<br>`03_web_app.py` ← `tutorials/04_product/03_web_app.py`<br>`04_monitoring.py` (신규: MLflow Tracing으로 대화 흐름 추적, 아래) | f7 |
| `appendix` | — | 선택 실습 | `embedding_speed.py` ← `tutorials/03_optimize/01_embedding_speed.py`<br>`embedding_exploration.py` ← `tutorials/01_ds_core/appendix_exploration.py`<br>`gemini_basics.py` ← `tutorials/01_ds_core/appendix_gemini_basics.py` | f7 |

- `tutorials/_legacy/`(에이전트·모니터링): MLflow 모니터링이 들어오면 모니터링은 대체된다. 에이전트는 PLAN 미정 사항이 정해질 때까지 유지하다가 옮기거나 지운다.
- 옮길 때 `git mv`로 이력을 남기고, README·PLAN·ARCHITECTURE·각 앱 README의 경로를 함께 고친다.

## MLflow를 어디에 붙이나 (구현은 별도 설계 후)

사용자 요청(2026-09-30): MLflow 3.x로 실험 추적 · 모델 레지스트리 · 서비스 모니터링(Tracing으로 대화 흐름까지).

| 기능 | 붙는 곳 | 실습 |
|---|---|---|
| 실험 추적 | `ragkit train` · `evaluate` · `compare` · `ragkit-bench run`이 파라미터·지표·산출물을 기록 | 02·03(평가·학습), 04(비교표) |
| 모델 레지스트리 | 파인튜닝 · ONNX · INT8 · 가지치기 모델을 버전 등록, API·CLI가 별칭(예: `champion`)으로 불러옴 | 03 → 04 |
| Tracing / 서비스 모니터링 | api·search-cli·mcp의 요청 한 건 = 트레이스(검색 → LLM 스팬, 토큰·지연), 노트북 대화 = 세션으로 묶음 | 05 |

## 순서

1. MLflow 설계 확정 → 구현 (f7)
2. 폴더 이동과 경로 수정 (f7), 42 세션에 알림
3. 00·04·05 `SLIDE.md` 작성과 스크립트 두 모드 정리 (f7) / 01~03 (42)
4. 1~2교시 이론 장표 (사용자)
