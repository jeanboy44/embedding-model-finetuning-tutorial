# 강의 계획: 임베딩 모델 파인튜닝

부제: AI 시대 생존법: 내 업무 확장편

> 이 저장소의 모든 자료(튜토리얼, 스크립트, 문서)는 이 흐름을 따른다.
> 새 작업을 시작하는 에이전트는 이 파일을 먼저 읽고, 만드는 것이 아래 4단계 중 어디에 속하는지 확인한다.
> 1단계는 `ragkit`(라이브러리), 2~4단계는 `apps/*`(ragkit을 쓰는 앱)에 만든다.

## 핵심 메시지

데이터 사이언티스트(DS)의 본업은 모델을 고르고, 학습하고, 평가하는 일이다.
예전에는 그 모델을 "쓸 수 있게" 만드는 일(API, 배포 최적화, 도구, 화면)은 다른 직군의 몫이거나, DS가 억지로 떠안는 부담이었다.
AI 코딩 도구가 오면서 DS 한 사람이 그 범위까지 직접, 빠르게 해낼 수 있게 됐다.
강의는 **같은 임베딩 모델 하나를 들고 DS의 업무 범위가 단계별로 넓어지는 과정**을 보여 준다.

```
[1. DS 본업]      모델 비교 → 학습 → 평가                  ragkit (src/ragkit/)
      ↓                                                      ▲ import
[2. +α 업무]      API로 감싸서 남이 쓰게 하기             apps/api
      ↓  ── AI가 오면서 ──
[3. 배포 최적화]  ONNX 변환 · 양자화 · 속도/정확도 비교   ragkit(변환) + apps/bench(비교)
      ↓
[4. 제품화]       CLI 도구 · MCP · React 프론트엔드        apps/search-cli, apps/mcp(제안), apps/web
```

주제 도메인은 대한민국 법령 조문 검색이다(데이터 결정은 `docs/handoff-law-question-gen.md` 참고).

## 교시 구성 (2026-09-30 확정)

8교시. 교시마다 50분 수업 + 10분 휴식(2026-10-01 결정). 교시마다 **그 시간에 하는 업무를 설명하는 장표 1개 + 그 업무를 직접 실행하는 스크립트**가 짝이다.
실습은 "2026년의 내가 하는 업무를 같이 해 본다"는 설정이고, 코드를 한 줄씩 보기보다 모델 개발 + 배포의 전체 과제 흐름을 수행해 보는 것이 목적이다.
실습 장표에는 그 업무의 2022년 모습과 2026년 모습을 한 줄씩 넣는다.

| 교시 | 구분 | 업무 / 내용 |
|---|---|---|
| 1 | 이론 | 내 소개: DS가 하는 일, AI 도입 전후(2022 vs 2026)의 현업 변화 |
| 2 | 이론 | 기술 용어·개념(ML/DL, 임베딩, 파인튜닝, 백엔드/API, CLI, MCP, 프론트엔드) → 강의 의도(왜 임베딩 파인튜닝인가) → 필요성(검색 누락이 RAG 정확도 상한, 쿼리 확장은 질문마다 LLM 비용) → 주제(법령이 적절한 사례인 이유: 일상어 vs 법률 용어, 통째로 못 넣는 코퍼스) → 실습 지도(실습 0~5) |
| 3 | 실습 0 | 환경 설정 + 완성품 먼저 써 보기(웹·CLI·MCP, base vs 파인튜닝 결과 나란히) + RAG 비교로 필요성 확인 |
| | **2022의 업무** | **DS 본업: 모델을 만든다** |
| 4 | 실습 1 | 데이터 만들기: 코퍼스, 질문 생성(Claude 스킬 — "같은 일도 방식이 바뀌었다"의 예), 법령 단위 분할 |
| 5 | 실습 2 | 평가: 평가셋과 지표 → **학습 없이 쓸 수 있는 선택지 비교**: 베이스 모델 비교(실험 005) + LLM 쿼리 확장(실험 003, 질문당 LLM 비용) → "학습해야 하는가"의 근거 |
| 6 | 실습 3 | 학습 실험과 분석: **DS 관점의 실험 iteration**(가설 → 설정 → 파일럿 → 학습 → dev로 선택 → test 1회 → 분석 → 다음 가설)과 **결과 분석**(실험 간 비교, 질문 유형·테마별, 실패 사례, 파인튜닝의 한계)에 초점. 학습 원리(MNRL, NO_DUPLICATES, hard negative, 전체 vs LoRA)는 짧게. 학습은 몇 step 맛보기만, 강사가 학습한 모델을 나눠 준다. **AI 시대에 더 쉽게 하는 방법**(AI에게 실험 설정·분석 스크립트·실패 사례 분류·리포트를 맡기고 DS는 가설과 판단에 집중, MLflow로 실험 비교) |
| | **2026의 업무** | **확장: 모델을 전달한다** |
| 7 | 실습 4 | 서빙과 배포 최적화: API는 수강생이 이미 배웠으므로 짧게 복습 → ONNX, INT8, 어휘 가지치기, 비교표 |
| 8 | 실습 5 | 제품화(CLI·MCP·웹), 마무리(다시 2022 vs 2026) |

실습 2 비교표 (학습 없이 쓸 수 있는 선택지. 실험 003 `experiments/exp_003_llm_query_expansion`):

| 조합 | 지표 | 질문당 LLM 호출 |
|---|---|---|
| e5 base | R@k, MRR, nDCG | 0 |
| e5 base + 쿼리 확장 | | 1 |
| EmbeddingGemma base (더 큰 모델) | | 0 |

실습 3 비교표 (실험 007, `feat/finetune-runs`): base e5 / Gemma / 002 전체 학습 / 004 LoRA / 006 배치 128. 실습 2 표의 쿼리 확장 행을 옆에 두고 "파인튜닝은 질문당 LLM 비용을 학습 1회로 옮긴다"를 확인한다.

운영 원칙: 무거운 산출물(모델, 파인튜닝 모델, 인덱스, 질문, split, 확장 쿼리 캐시)은 미리 만들어 Drive로 배포하고, 스크립트는 "받은 산출물로 확인" 모드와 "작은 부분집합으로 직접 실행" 모드를 둔다.

### 다음 작업 (2026-09-30)

1. 파인튜닝 재학습 — 지금 `models/finetuned/exp_002`는 옛 split으로 학습해 새 test(2,526개)와 겹칠 수 있다. 새 split으로 002·004·006 재학습 후 실험 007 (`feat/finetune-runs`)
2. 쿼리 확장 평가 — 코드 완료(`ragkit expand`, `evaluate --expand`, 실험 003 설정). gemini-2.5-flash-lite 무료 등급은 하루 20회라 실행 보류(2026-09-30), 캐시 17/2,526. 데이터 확보는 사용자 결정으로 미룸(2026-10-01). 실습 2 자료는 확장 행을 비워 두고 만든다
3. 위 두 비교표 채우기
4. 교시 기준으로 `lecture/` 재구성 (`docs/HANDSON_PLAN.md`). 폴더 이름은 `tutorials` → `lecture`로 바꿨다(2026-10-01)

## 저장소 구조

경계 기준: 모델·데이터를 **만드는** 것은 ragkit, 만든 것을 **쓰는** 것은 apps.

```
pyproject.toml         # 루트 = ragkit 패키지 + uv workspace (members = apps/api, apps/bench, apps/search-cli, apps/mcp)
src/ragkit/            # 1단계. data · training · evaluation · embeddings · retrieval · rag · export-onnx · 양자화 변환
apps/
  api/                 # 2단계. FastAPI 검색 API (ragkit + fastapi)
  bench/               # 3단계. 원본 / ONNX / 양자화 모델 비교표
  search-cli/          # 4단계. 최종 사용자용 검색 CLI (ragkit core만, uvx 배포)
  mcp/                 # 4단계. MCP 서버 (Claude 등 에이전트가 조문 검색)
  web/                 # 4단계. NotebookLM형 React 웹앱 '법령 노트' (Node 프로젝트, workspace 멤버 아님, apps/api만 호출)
experiments/           # 코드 없음. 실험별 config.yaml + 결과
lecture/               # 강의 자료. 실습 스크립트(ragkit과 apps를 호출만 한다) + slides/(HTML 슬라이드)
  01_ds_core/          # 1단계
  02_api/              # 2단계
  03_optimize/         # 3단계
  04_product/          # 4단계
scripts/               # 일회성 데이터 준비 (prepare_law_data, 질문 Drive 업로드·다운로드, 모델 다운로드)
```

- apps는 ragkit의 **공개 함수만** 쓴다(모델 로드, embed, 인덱스 생성·로드, search, evaluate_retrieval). 내부 모듈을 import하지 않는다.
- `ragkit` CLI(split / train / evaluate / export-onnx)는 DS용 도구라 ragkit에 둔다. 4단계 CLI는 최종 사용자용이라 별도 앱이다.
- 설치: 강의는 `uv sync --all-packages --all-extras`, 앱 실행은 `uv run --package <앱> ...`.
- 패키지 이름: `ragkit`(CLI `ragkit`), `ragkit-api`, `ragkit-bench`, `ragkit-search`(진입점 `ragkit-search`), `ragkit-mcp`.
- 의존성: ragkit core는 onnxruntime + tokenizers(배포 기본 ONNX), torch/transformers는 extra `[torch]`, 학습용은 extra `[train]`.

## 단계별 내용

### 1단계. DS 본업: 모델 비교, 학습, 평가 (ragkit)

기존 DS로서 해 왔던 일. 강의의 기준점이다.

- 도입(왜 파인튜닝인가): 코퍼스(약 700만 자)는 LLM에 통째로 넣을 수 없다 → 에이전트형 탐색 vs 임베딩 RAG → 검색에서 놓친 조문은 답할 수 없으니 검색 누락이 정확도 상한이 된다 → 쿼리 확장으로 메우면 쿼리마다 LLM 비용이 늘어 주객이 바뀐다 → 파인튜닝으로 그 비용을 학습 1회로 옮긴다
  - 학습 데이터 생성: 운영이라면 저가 API, 개발은 SKILL.md, 강의에서는 skill 약식으로 한다
  - 비교 실습: 순수 LLM / RAG / RAG+쿼리 확장
- 베이스 모델 비교: 파인튜닝 대상 `intfloat/multilingual-e5-small`과 비교 대상 `google/embeddinggemma-300m`을 같은 평가셋으로 비교 (`docs/adr/001-model-selection.md`)
- 데이터 준비: legalize-kr 법령 코퍼스 (`scripts/prepare_law_data.py`, 긴 조문은 항·호 단위로 나눠 약 2.6만 문서), Claude 스킬로 질문·정답 조문·hard negative 생성 (`.claude/skills/law-question-gen/`, 실습 폴더 `lecture/01_data/README.md`는 스킬 사용법만 안내), 질문 공유는 `scripts/law_questions_drive.py`
- 분할: 법령 단위(확정). 테마마다 법령 ~20%는 test, ~10%는 dev. 학습 때 본 적 없는 법령에서도 좋아지는지 측정
- 파인튜닝: 전체 학습(`exp_002_finetuned`)과 LoRA(`exp_004_lora`)를 따로 실험. MNRL + `BatchSamplers.NO_DUPLICATES`, e5 접두어는 코드에서 붙인다
- 평가: 전체 코퍼스 대상 Recall@k, MRR, nDCG로 base / 전체 학습 / LoRA (필요하면 쿼리 확장도) 비교
- 만능 아님: 추론, 수치 필터, 코퍼스에 없는 지식은 파인튜닝으로 고쳐지지 않는다
- 관련 위치: `ragkit.data`, `ragkit.training`, `ragkit.evaluation`, `ragkit` CLI, `experiments/`

### 2단계. +α로 떠안던 일: API 개발 (apps/api)

모델을 만든 뒤 "그래서 어떻게 써요?"라는 요청에 답하려고 DS가 추가로 해야 했던 일.

- 1단계 모델을 검색 API로 감싸기: FastAPI `POST /search` → 상위 k개 조문 반환
- 요청/응답 스키마, 시작 시 모델·인덱스 로딩, 헬스체크
- 처음에는 torch 백엔드로 띄운다(3단계에서 ONNX로 바꾸는 전후를 비교하기 위해)
- 메시지: 예전에는 이 단계가 DS에게 낯설고 시간이 많이 드는 일이었다

### 3단계. AI가 오면서: 배포 최적화 (ragkit 변환 + apps/bench)

AI 도구 덕분에 DS가 직접 손대기 쉬워진 영역 ①.

- `ragkit export-onnx`로 PyTorch → ONNX 변환, 양자화(동적 INT8, 채널별)
- 어휘 가지치기(`ragkit prune-vocab`): 용량의 82%인 다국어 어휘 임베딩 표를 한국어 법령에 필요한 토큰(25만 → 약 1.9만)만 남긴다
- `apps/bench`: 원본 / ONNX / INT8 / 가지치기+INT8의 **지연 시간·메모리·모델 크기·설치 크기·검색 정확도(Recall@k)** 비교표
- `apps/api`의 백엔드를 torch → ONNX로 바꿔 설치 크기·시작 시간을 비교한다 (Docker는 쓰지 않는다)
- 메시지: "정확도를 얼마나 잃고 얼마나 빨라지는가"를 DS가 직접 재고 판단한다

### 4단계. AI가 오면서: CLI 도구, MCP, 프론트엔드 (apps/search-cli, apps/mcp, apps/web)

> MCP를 4단계에 두는 것은 제안이다(apps에 넣는 것은 확정).

AI 도구 덕분에 DS가 직접 손대기 쉬워진 영역 ②. 모델을 "제품"으로 전달한다.

- `apps/search-cli`: 동료가 설치해서 바로 쓰는 검색 도구 (cyclopts, `uvx`로 실행, ragkit core + ONNX만)
- `apps/mcp`: MCP 서버로 Claude 같은 에이전트가 조문을 검색하게 한다
- `apps/web`: NotebookLM형 '법령 노트'. 노트북 = 법령 묶음, 그 안에서 인용 달린 답을 스트리밍, 인용을 누르면 조문 원문, 답을 노트로 저장 (2단계 API만 호출). 스택: Vite + React + TS + Tailwind + shadcn/ui + TanStack Query (2026-09-29 확정)
- 네 앱 모두 ragkit의 `Searcher`(`ragkit.service`)를 입구로 쓴다 (구조는 `docs/ARCHITECTURE.md`)
- MLflow 3 모니터링 (`ragkit.tracking`, 2026-10-01): 실험 추적(train·evaluate·compare·bench run) → 모델 레지스트리(`ragkit register`, 앱은 `models:/law-embedder@champion`) → 서비스 트레이싱(질문 한 번 = 트레이스, 검색 조문·프롬프트·토큰, 노트북 = 세션)
- 메시지: DS 한 사람이 모델부터 사용자 화면까지 끝까지 만든다

## 진행 현황 (2026-09-30)

| 단계 | 소스 | 튜토리얼 | 확인 |
|---|---|---|---|
| 1 | ragkit(인덱스·분할·학습·평가·비교) | `01_build_rag.py`만 | 질문 12,232개 → 법령 단위 분할 train 8,294 · dev 1,412 · test 2,526(법령 16개). 실험 005: e5-small R@5 0.513 / EmbeddingGemma 0.743. 파인튜닝은 `feat/finetune-runs`에서 진행 중(새 분할로 재학습 필요). 비교 실습 02~05와 학습·평가 튜토리얼 없음 |
| 2 | apps/api | `01_search_api.py`, `02_streaming_and_notebooks.py` | 실행 확인 (torch 백엔드, SSE, 노트북) |
| 3 | `ragkit quantize`(채널별 INT8), `ragkit prune-vocab`, apps/bench | `01`~`03` | 실험 008(test 2,526개): 가지치기+INT8이 torch 대비 설치 669→127MB·모델 471→30MB·메모리 1133→403MB·로딩 3.6→0.3s, R@5 0.513→0.515(동일). 가지치기 어휘가 test 질문 100%를 원본과 같게 토큰화 |
| 4 | apps/search-cli·mcp·web, `ragkit.tracking`(MLflow) | `01_search_cli.py`, `02_mcp_server.py`, `03_web_app.py`, `04_monitoring.py` | uvx(torch 없음), MCP stdio 도구 호출, 웹 흐름(Playwright) 확인. MLflow(`feat/mlflow`): bench run 2개 → 레지스트리 champion으로 API 기동 → 노트북 대화 2턴이 세션 하나의 트레이스 2개(검색 조문·토큰 수)로 남는 것 확인 |

- 이전 구성(Phase 1~4)의 자료는 모두 지웠다(2026-10-01). 옛 `tutorials/_legacy`(에이전트·모니터링)와 그것만 쓰던 `ragkit.monitoring`, `DocumentStore`·`retrieve`, `run_rag`, `config.EXPERIMENTS`까지 정리했다.

## 미정 사항

- (결정) 시간 배분은 위 "교시 구성"을 따르고, 교시마다 50분 수업 + 10분 휴식 (2026-10-01)
- (결정) RAG 에이전트는 지웠다 (2026-10-01). 2교시에 "에이전트형 탐색 vs 임베딩 RAG"를 한 줄로 넣고, 에이전트 흐름은 실습 5의 MCP가 보여 준다
- (결정) 옛 모니터링은 지우고 MLflow(실습 5)로 대체 (2026-10-01)
- (결정) 배포는 Docker 없이 `ragkit-api --web-dist apps/web/dist`(한 프로세스)로 한다 (2026-09-30)
