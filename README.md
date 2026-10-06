# 임베딩 모델 파인튜닝 — AI 시대 생존법: 내 업무 확장편

같은 임베딩 모델 하나를 들고, 데이터 사이언티스트(DS)의 업무 범위가 모델 개발에서 API·배포 최적화·제품화까지 넓어지는 과정을 따라가는 8시간 실습 강의 저장소입니다. 주제 도메인은 대한민국 법령 조문 검색입니다.

## 강의 구성

| 단계 | 내용 | 코드 | 실습 |
|---|---|---|---|
| 1. DS 본업 | 왜 파인튜닝인가(RAG 비교) → 데이터 준비 → 평가 → 학습 | `src/ragkit/` | `lecture/03_setup` ~ `06_train` (실습 0~3) |
| 2. +α 업무 | 모델을 검색 API로 감싸기 | `apps/api/` | `lecture/07_optimize/01_api_review.py` (실습 4 복습) |
| 3. 배포 최적화 | ONNX 변환 · INT8 양자화 · 어휘 가지치기 · 속도/메모리/정확도 비교 | `ragkit` + `apps/bench/` | `lecture/07_optimize/` (실습 4) |
| 4. 제품화 | 검색 CLI · MCP 서버 · React 화면 · MLflow 모니터링 | `apps/search-cli/`, `apps/mcp/`, `apps/web/`, `ragkit.tracking` | `lecture/08_product/` (실습 5) |

강의는 8교시(이론 2 + 실습 0~5)이고, `lecture/`의 폴더 하나가 교시 하나입니다(`01_introduction` ~ `08_product`). 교시마다 장표(claude.ai Slides 덱)와 스크립트 묶음이 짝입니다. 교시별 폴더와 실행 규칙은 [lecture/README.md](lecture/README.md)를 보세요.

자세한 흐름과 결정 사항은 [강의 계획](docs/PLAN.md)을 참고하세요.

## 빠른 시작

### 설치

- [macOS 설치 가이드](INSTALL_MAC.md) · [Windows 설치 가이드](INSTALL_WINDOWS.md)

```bash
uv sync --all-packages --all-extras   # 강의용: ragkit + 모든 앱 + 학습 도구
```

배포용 최소 설치는 `ragkit` core(ONNX 추론)만 설치합니다. torch는 extra `[torch]`, 학습·변환 도구는 extra `[train]`입니다.

### 모델과 데이터 준비

기본 모델은 한국어를 지원하는 [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)(MIT 라이선스)입니다.

```bash
uv run python scripts/download_model_hf.py               # 모델 (또는 scripts/download_model_gdrive.py)
uv run ragkit export-onnx models/multilingual-e5-small   # ONNX 변환 (배포 기본 백엔드)
uv run ragkit quantize models/multilingual-e5-small      # INT8 양자화 → models/multilingual-e5-small-int8
uv run ragkit prune-vocab models/multilingual-e5-small   # 어휘 가지치기 → -pruned (이어서 export-onnx, quantize)
uv run python scripts/prepare_law_data.py                # 법령 코퍼스 → data/processed/law_docs.json
uv run ragkit index                                      # 검색 인덱스 → data/processed/index/<모델>.sqlite
uv run ragkit split data/questions                       # 평가 질문 분할 → data/splits/ (질문은 Drive에서 받는다)
```

LLM 단계(답변 생성)는 `.env`에 `GEMINI_API_KEY`가 필요합니다. [Google AI Studio](https://aistudio.google.com)의 무료 등급 키로 충분합니다(기본 모델 `gemini-2.5-flash-lite`). 키가 없으면 해당 단계만 건너뜁니다.

## 실습 스크립트

모두 저장소 루트에서 `uv run python <파일>`로 실행합니다. 기본은 받은 산출물로 확인하고, `--run`을 주면 작은 부분집합으로 직접 실행합니다.

| 실습 (교시) | 파일 | 내용 |
|---|---|---|
| 0 (3) | `lecture/03_setup/01_check_env.py` | 설치·모델·데이터·인덱스·API 키 점검, 빠진 것을 받는 명령 |
| 0 (3) | `lecture/03_setup/02_try_product.py` | 같은 질문을 학습 전 e5 vs 파인튜닝 모델로 나란히, 제품 띄우는 법 |
| 0 (3) | `lecture/03_setup/03_why_rag.py` | 통째로 넣으면? → 인덱스 → 검색 → 조문 근거 답변 |
| 1 (4) | `lecture/04_data/01_corpus.py` | 코퍼스 통계, 항·호 분할, 길이 분포 |
| 1 (4) | `lecture/04_data/02_questions.py` | 질문 유형·테마, hard negative, 검증 스크립트 (질문 생성은 Claude 스킬, `lecture/04_data/README.md`) |
| 1 (4) | `lecture/04_data/03_split.py` | 법령 단위 분할과 왜 법령 단위인가 |
| 2 (5) | `lecture/05_evaluate/01_metrics.py` | R@k · MRR · nDCG · doc/article/multi 판정, 학습 전 e5 test 평가 |
| 2 (5) | `lecture/05_evaluate/02_query_expansion.py` | LLM 쿼리 확장: 정확도와 질문당 비용 (받은 캐시) |
| 3 (6) | `lecture/06_train/01_train_taste.py` | MNRL · in-batch negative · NO_DUPLICATES, 학습 기록, 몇 step 맛보기 |
| 3 (6) | `lecture/06_train/02_compare_runs.py` | 실험 010 비교표, 가설 판정, paired-test |
| 3 (6) | `lecture/06_train/03_error_analysis.py` | 고쳐진·새로 틀린·여전히 틀리는 질문, 파인튜닝의 한계 |
| 4 (7) | `lecture/07_optimize/01_api_review.py` | API 복습: 로딩, search·필터·422, SSE 스트리밍, 노트북 |
| 4 (7) | `lecture/07_optimize/02_onnx_quantize_prune.py` | ONNX 변환, INT8 텐서 단위 vs 채널별, 어휘 가지치기 |
| 4 (7) | `lecture/07_optimize/03_bench_and_serve.py` | 원본/ONNX/INT8/가지치기 비교표, API 백엔드 교체 |
| 5 (8) | `lecture/08_product/01_search_cli.py` | 검색 CLI, `--json`, uvx 배포 |
| 5 (8) | `lecture/08_product/02_agent_skill.py` | 에이전트 스킬(SKILL.md)로 Claude Code·Gemini CLI에 CLI 붙이기, 학습 전 vs 파인튜닝 비교 |
| 5 (8) | `lecture/08_product/03_mcp_server.py` | MCP: 연결 → 도구 목록 → 호출, Claude 등록 |
| 5 (8) | `lecture/08_product/04_web_app.py` | 웹앱 빌드와 API 서버 한 주소 배포 |
| 5 (8) | `lecture/08_product/05_monitoring.py` | MLflow: 실험 run 비교 → 모델 레지스트리(champion) → 서비스 트레이스·세션 |
| 선택 | `lecture/02_concepts/01_embedding_exploration.py` · `03_setup/extra_gemini_basics.py` · `07_optimize/extra_embedding_speed.py` | 임베딩 공간 탐색, Gemini 기초, 임베딩 속도 옵션 |

## 도구와 앱

```bash
# DS용 CLI (1·3단계): index · split · train · evaluate · compare · export-onnx · quantize · prune-vocab
uv run ragkit --help

# 배포 최적화 비교표 (3단계) → experiments/exp_008_deploy_bench/results/comparison.md
uv run ragkit-bench run

# 검색 API (2단계) → http://127.0.0.1:8000/docs
uv run --package ragkit-api ragkit-api --backend torch                              # 2단계
uv run --package ragkit-api ragkit-api --model models/multilingual-e5-small-pruned-int8    # 3단계 이후
uv run --package ragkit-api ragkit-api --web-dist apps/web/dist                            # 화면까지 한 주소

# 제품 (4단계)
uv run --package ragkit-search ragkit-search search "야간 근로 수당" --law 근로기준법
uv run --package ragkit-mcp ragkit-mcp                      # Claude 등록은 apps/mcp/README.md
cd apps/web && pnpm install && pnpm dev                     # 개발 모드 (api를 먼저 띄운다)

# MLflow (4단계): .env에 MLFLOW_TRACKING_URI가 있으면 train·evaluate·compare·bench가 run을,
# api·search-cli·mcp가 질문마다 트레이스를 남긴다 (없으면 아무것도 기록하지 않음)
uv sync --extra mlflow
uv run mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --artifacts-destination mlruns/artifacts --port 5050  # .env: MLFLOW_TRACKING_URI=http://127.0.0.1:5050 (macOS는 5000을 AirPlay가 씀)
uv run ragkit register models/multilingual-e5-small-pruned-int8 --alias champion   # → models:/law-embedder@champion
uv run --package ragkit-api ragkit-api --model models:/law-embedder@champion

# 데이터 버전: 코퍼스 + 분할 + 원본 질문을 v1, v2, …로 고정한다.
# 파일은 Drive의 law-data-<버전>.zip, git에는 manifest(data/versions/<버전>.json)만 둔다.
uv run python scripts/data_version.py pull v1        # Drive에서 받아 data/에 풂 (로그인 불필요)
uv run python scripts/data_version.py status         # 지금 data/가 어느 버전과 같은지
# 새 버전 만들기 (강사): create → upload(rclone) → register(MLflow law-data 실험)
uv run python scripts/data_version.py create v2 --description "무엇이 바뀌었는지"
```

train·evaluate·compare run에는 입력 파일과 내용이 같은 버전이 데이터셋(`law-retrieval/train` 등)과
`data_version` 태그로 남는다. 버전에 없는 파일로 돌리면 `data_version=unversioned`.

| 앱 | 설명 |
|---|---|
| [`apps/api`](apps/api) | `/api/search`, `/api/answer(/stream)`, `/api/notebooks/...` |
| [`apps/bench`](apps/bench) | `ragkit-bench run`: 지연·처리량·메모리·모델/설치 크기·Recall |
| [`apps/web`](apps/web/README.md) | 노트북 = 법령 묶음, 인용 달린 답, 조문 보기, 노트 |
| [`apps/search-cli`](apps/search-cli/README.md) | `search` · `ask` · `laws` · `show` · `skill install`(에이전트 스킬) |
| [`apps/mcp`](apps/mcp/README.md) | `search_laws` · `get_article` · `list_laws` · `ask` |

## 저장소 구조

```
src/ragkit/      # 1단계 라이브러리: data · embeddings · retrieval · rag · service · training · evaluation · CLI
apps/            # 2~4단계: ragkit을 쓰는 앱 (api·bench·search-cli·mcp는 uv workspace 멤버, web은 pnpm)
experiments/     # 실험 설정(config.yaml)과 결과(results/, git 제외)
lecture/         # 강의 자료: 교시별 폴더 8개(01_introduction ~ 08_product)
scripts/         # 일회성 데이터·모델 준비
docs/            # 강의 계획, 아키텍처, ADR
```

## 문서

- [강의 계획](docs/PLAN.md)
- [아키텍처](docs/ARCHITECTURE.md) · [ADR](docs/adr/)
