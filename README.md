# 임베딩 모델 파인튜닝 — AI 시대 생존법: 내 업무 확장편

같은 임베딩 모델 하나를 들고, 데이터 사이언티스트(DS)의 업무 범위가 모델 개발에서 API·배포 최적화·제품화까지 넓어지는 과정을 따라가는 8시간 실습 강의 저장소입니다. 주제 도메인은 대한민국 법령 조문 검색입니다.

## 강의 구성

| 단계 | 내용 | 코드 | 튜토리얼 |
|---|---|---|---|
| 1. DS 본업 | 왜 파인튜닝인가(RAG 비교) → 데이터 준비 → 학습 → 평가 | `src/ragkit/` | `tutorials/01_ds_core/` |
| 2. +α 업무 | 모델을 검색 API로 감싸기 | `apps/api/` | `tutorials/02_api/` |
| 3. 배포 최적화 | ONNX 변환 · INT8 양자화 · 어휘 가지치기 · 속도/메모리/정확도 비교 | `ragkit` + `apps/bench/` | `tutorials/03_optimize/` |
| 4. 제품화 | 검색 CLI · MCP 서버 · React 화면 | `apps/search-cli/`, `apps/mcp/`, `apps/web/` | `tutorials/04_product/` |

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

## 튜토리얼

모두 `uv run python <파일>`로 실행합니다.

| 단계 | 파일 | 내용 |
|---|---|---|
| 1 | `tutorials/01_ds_core/01_build_rag.py` | 통째로 넣으면? → 인덱스 → 검색 → 조문 근거 답변 |
| 2 | `tutorials/02_api/01_search_api.py` | 시작 시 로딩, health/search/법령 필터/검증(422)/answer (torch 백엔드) |
| 2 | `tutorials/02_api/02_streaming_and_notebooks.py` | SSE 스트리밍 답변, 노트북 → 대화 → 노트 저장 |
| 3 | `tutorials/03_optimize/01_embedding_speed.py` | 인덱싱 속도 옵션 (장치·길이순 배치·스레드) |
| 3 | `tutorials/03_optimize/02_onnx_and_quantize.py` | ONNX 변환, INT8 텐서 단위 vs 채널별, 어휘 가지치기 |
| 3 | `tutorials/03_optimize/03_bench_and_deploy.py` | 원본/ONNX/INT8/가지치기 비교표, API 백엔드 교체 |
| 4 | `tutorials/04_product/01_search_cli.py` | 검색 CLI, `--json`, uvx 배포 |
| 4 | `tutorials/04_product/02_mcp_server.py` | MCP: 연결 → 도구 목록 → 호출, Claude 등록 |
| 4 | `tutorials/04_product/03_web_app.py` | 웹앱 빌드와 API 서버 한 주소 배포 |

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
```

| 앱 | 설명 |
|---|---|
| [`apps/api`](apps/api) | `/api/search`, `/api/answer(/stream)`, `/api/notebooks/...` |
| [`apps/bench`](apps/bench) | `ragkit-bench run`: 지연·처리량·메모리·모델/설치 크기·Recall |
| [`apps/web`](apps/web/README.md) | 노트북 = 법령 묶음, 인용 달린 답, 조문 보기, 노트 |
| [`apps/search-cli`](apps/search-cli/README.md) | `search` · `ask` · `laws` · `show` |
| [`apps/mcp`](apps/mcp/README.md) | `search_laws` · `get_article` · `list_laws` · `ask` |

## 저장소 구조

```
src/ragkit/      # 1단계 라이브러리: data · embeddings · retrieval · rag · service · training · evaluation · CLI
apps/            # 2~4단계: ragkit을 쓰는 앱 (api·bench·search-cli·mcp는 uv workspace 멤버, web은 pnpm)
experiments/     # 실험 설정(config.yaml)과 결과(results/, git 제외)
tutorials/       # 단계별 실습 (01_ds_core ~ 04_product)
scripts/         # 일회성 데이터·모델 준비
docs/            # 강의 계획, 아키텍처, ADR
```

## 문서

- [강의 계획](docs/PLAN.md)
- [아키텍처](docs/ARCHITECTURE.md) · [ADR](docs/adr/)
