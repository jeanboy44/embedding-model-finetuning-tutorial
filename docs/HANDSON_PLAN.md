# 강의 자료 재구성 계획 (lecture/)

상태: **완료** (2026-10-06). 같은 날 저녁 폴더를 교시 번호 8개(`01_introduction` ~ `08_product`)로 다시 바꿨다(아래 "폴더 규칙"). 표의 폴더 이름은 그 전 이름이다. 교시 폴더로 옮기고(`git mv`) 실습 0~3 스크립트를 새로 썼다. 모든 스크립트는 기본 모드와 `--run`으로 끝까지 실행해 확인했다. 폴더 규칙과 스크립트 작성 규칙은 `lecture/README.md`. 아래 표의 "← 기존 파일"은 옮기기 전 위치다. 교시 구성은 `docs/PLAN.md` "교시 구성"을 따른다.

## 폴더 규칙

```
lecture/
  01_introduction/  # 1교시 내 소개 — 장표를 내보낸 HTML 하나만
  02_concepts/      # 2교시 용어·개념
  03_setup/         # 3교시 실습 0 (선택: Gemini 기초)
  04_data/          # 4교시 실습 1
  05_evaluate/      # 5교시 실습 2
  06_train/         # 6교시 실습 3
  07_optimize/      # 7교시 실습 4 (선택: 임베딩 속도 옵션)
  08_product/       # 8교시 실습 5
```

- (2026-10-06 저녁 변경) 폴더 번호 = 교시 번호, 폴더는 8개. 옛 `appendix/`는 관련 교시의 `extra_*.py`로 흩었고, 옛 HTML 장표 `slides/`는 Slides 덱으로 옮겼으므로 지웠다(git 이력에 남음)
- 교시 폴더에는 번호 붙은 실행 스크립트 `01_*.py`, `02_*.py` …를 둔다. 선택 실습은 `extra_*.py`
- (2026-10-06 변경) 장표는 교시마다 claude.ai Slides 덱 하나로 만든다. 목록은 `lecture/README.md`. 아래는 그 전의 HTML 규칙이다.
- 장표는 `lecture/slides/`에 교시마다 HTML 한 파일로 둔다(예: `03_train.html`, 2026-10-01 결정, 처음 계획의 `SLIDE.md`를 대신함). 공용 `assets/theme.css`·`deck.js`를 불러 쓰고, 맨 앞에 **2022년의 이 업무 / 2026년의 이 업무**를 한 줄씩, 이어서 목표·흐름·실행할 스크립트·확인할 결과
- 슬라이드 모양: 최대한 심플, 옅은 그라데이션 배경, Pretendard, **2022 = 주황 · 2026 = 파랑**, 1600x900 무대를 한 장씩 넘김(←→·스페이스·클릭, F 전체 화면). 예시는 `lecture/slides/course-scope.html`
- 스크립트는 두 모드를 둔다: 기본은 **"받은 산출물로 확인"**(Drive로 배포한 모델·인덱스·질문·split 사용), `--run`이면 **"작은 부분집합으로 직접 실행"**
- 스크립트 머리 docstring 형식은 지금 튜토리얼과 같다(학습 목표 · 사전 준비 · 실행)
- 실행은 저장소 루트에서 `uv run python lecture/<폴더>/<파일>.py`

## 교시별 내용과 기존 파일 매핑

| 폴더 | 교시 | 업무 | 스크립트 (← 기존 파일) | 담당 |
|---|---|---|---|---|
| `00_setup` | 3 | 환경 설정, 완성품 먼저 써 보기, 학습 전후 비교 | `01_doctor.py` (점검 · 동작 확인 · --fix)<br>`02_try_product.py` (질문 넷을 학습 전 vs 파인튜닝으로 나란히)<br>`README.md` (CLI 명령으로 학습 전후 비교, 2026-10-07 `03_why_rag.py`를 대신함) | f7 |
| `01_data` | 4 | 코퍼스, 질문 생성(Claude 스킬), 법령 단위 분할 | `README.md` (질문 생성은 `.claude/skills/law-question-gen` 스킬을 쓰라는 안내, 2026-10-01)<br>나머지 (신규) | 42 |
| `02_evaluate` | 5 | 평가셋·지표, 학습 없는 선택지 비교: 학습 전 e5 + LLM 쿼리 확장(실험 003). 강의는 e5로만 진행하므로 베이스 모델 비교(실험 005)는 뺐다(2026-10-06) | `01_metrics.py` · `02_query_expansion.py`(받은 확장 캐시 17개로 비교, `--run`은 키가 있으면 3개만 새로 확장) | 42 |
| `03_train` | 6 | DS 관점 실험 iteration과 분석(실험 간 비교, 유형·테마별, 실패 사례, 한계) + AI 시대에 더 쉽게 하는 법. 학습은 원리 짧게 + 맛보기, 받은 모델 사용 | `01_train_taste.py`(`--run`: `ragkit train --output-dir <임시> --max-steps 30`) · `02_compare_runs.py`(실험 010, paired-test) · `03_error_analysis.py`(r001_A, test) | 42 |
| `04_optimize` | 7 | API 짧은 복습 → ONNX · INT8 · 어휘 가지치기 · 비교표 | `01_api_review.py` ← `lecture/02_api/01_search_api.py` + `02_streaming_and_notebooks.py` (한 파일로 압축)<br>`02_onnx_quantize_prune.py` ← `lecture/03_optimize/02_onnx_and_quantize.py`<br>`03_bench_and_serve.py` ← `lecture/03_optimize/03_bench_and_deploy.py` | f7 |
| `05_product` | 8 | CLI · MCP · 웹, 운영 모니터링(MLflow), 마무리 | `01_search_cli.py` ← `lecture/04_product/01_search_cli.py`<br>`02_mcp_server.py` ← `lecture/04_product/02_mcp_server.py`<br>`03_web_app.py` ← `lecture/04_product/03_web_app.py`<br>`04_monitoring.py` (신규: MLflow Tracing으로 대화 흐름 추적, 아래) | f7 |
| `appendix` | — | 선택 실습 | `embedding_speed.py` ← `lecture/03_optimize/01_embedding_speed.py`<br>`embedding_exploration.py` ← `lecture/01_ds_core/appendix_exploration.py`<br>`gemini_basics.py` ← `lecture/01_ds_core/appendix_gemini_basics.py` | f7 |

- 이전 구성 자료(옛 `tutorials/_legacy`: 에이전트·모니터링)는 모두 지웠다(2026-10-01). 모니터링은 MLflow(05)가, 에이전트 흐름은 MCP(05)가 대신 보여 준다.
- 교시는 50분 수업 + 10분 휴식이다. 스크립트 실행 시간은 교시당 합계 30분 안쪽을 목표로 한다.
- 옮길 때 `git mv`로 이력을 남기고, README·PLAN·ARCHITECTURE·각 앱 README의 경로를 함께 고친다.

## MLflow를 어디에 붙이나 (2026-10-01 구현, `feat/mlflow`)

사용자 요청(2026-09-30): MLflow 3.x로 실험 추적 · 모델 레지스트리 · 서비스 모니터링(Tracing으로 대화 흐름까지).

| 기능 | 붙는 곳 | 실습 |
|---|---|---|
| 실험 추적 | `ragkit train` · `evaluate` · `compare` · `ragkit-bench run`이 파라미터·지표·산출물을 기록 | 02·03(평가·학습), 04(비교표) |
| 모델 레지스트리 | 파인튜닝 · ONNX · INT8 · 가지치기 모델을 버전 등록, API·CLI가 별칭(예: `champion`)으로 불러옴 | 03 → 04 |
| Tracing / 서비스 모니터링 | api·search-cli·mcp의 요청 한 건 = 트레이스(검색 → LLM 스팬, 토큰·지연), 노트북 대화 = 세션으로 묶음 | 05 |

쓰는 법: `.env`에 `MLFLOW_TRACKING_URI`만 넣으면 위 명령들이 코드 변경 없이 기록한다(없으면 no-op). 실습 02·03에서 쓰려면 `uv sync --extra mlflow` 후 `mlflow server`를 띄우고 같은 명령을 실행하면 된다. 05의 `04_monitoring.py`가 서버 기동 → bench run → `ragkit register` → 레지스트리 모델로 API → 노트북 대화 트레이스까지 한 번에 보여 준다(`--check`, 몇 분).

## 순서

1. ~~MLflow 설계 확정 → 구현 (f7)~~ 완료 (`feat/mlflow`, 2026-10-01)
2. ~~폴더 이동과 경로 수정~~ 완료 (2026-10-06)
3. ~~스크립트 두 모드 정리, 실습 0~3 스크립트~~ 완료 (2026-10-06). 장표는 1~8교시 모두 claude.ai Slides 덱으로 옮겼다(목록은 `lecture/README.md`). `lecture/slides/`의 HTML은 옮기기 전 버전이다
4. 1~2교시 이론 장표 (사용자)
