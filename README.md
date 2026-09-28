# 임베딩 모델 파인튜닝 — AI 시대 생존법: 내 업무 확장편

같은 임베딩 모델 하나를 들고, 데이터 사이언티스트(DS)의 업무 범위가 모델 개발에서 API·배포 최적화·제품화까지 넓어지는 과정을 따라가는 8시간 실습 강의 저장소입니다. 주제 도메인은 대한민국 법령 조문 검색입니다.

## 강의 구성

| 단계 | 내용 | 코드 | 튜토리얼 |
|---|---|---|---|
| 1. DS 본업 | 왜 파인튜닝인가(RAG 비교) → 데이터 준비 → 학습 → 평가 | `src/ragkit/` | `tutorials/01_ds_core/` |
| 2. +α 업무 | 모델을 검색 API로 감싸기 | `apps/api/` | `tutorials/02_api/` |
| 3. 배포 최적화 | ONNX 변환 · 양자화 · 속도/정확도 비교 | `ragkit` + `apps/bench/` | `tutorials/03_optimize/` |
| 4. 제품화 | 검색 CLI · MCP 서버 · React 화면 | `apps/search-cli/`, `apps/mcp/`, `apps/web/` | `tutorials/04_product/` |

자세한 흐름과 결정 사항은 [강의 계획](docs/PLAN.md)을 참고하세요.

## 빠른 시작

### 설치

- [macOS 설치 가이드](INSTALL_MAC.md) · [Windows 설치 가이드](INSTALL_WINDOWS.md)

```bash
uv sync --all-packages --all-extras   # 강의용: ragkit + 모든 앱 + 학습 도구
```

배포용 최소 설치는 `ragkit` core(ONNX 추론)만 설치합니다. torch는 extra `[torch]`, 학습 도구는 extra `[train]`입니다.

### 모델과 데이터 준비

기본 모델은 한국어를 지원하는 [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)(MIT 라이선스)입니다.

```bash
uv run python scripts/download_model_hf.py          # 모델 (또는 scripts/download_model_gdrive.py)
uv run ragkit export-onnx models/multilingual-e5-small   # ONNX 변환 (배포 기본 백엔드)
uv run python scripts/prepare_law_data.py           # 법령 코퍼스 → data/processed/law_docs.json
uv run ragkit index                                 # 검색 인덱스 → data/processed/index/<모델>.sqlite
```

LLM 단계(답변 생성, 쿼리 확장)는 `.env`에 `GEMINI_API_KEY`가 필요합니다. [Google AI Studio](https://aistudio.google.com)의 무료 등급 키로 충분합니다. 키가 없으면 해당 단계만 건너뜁니다.

### 실행 예제

```bash
# 1단계: 법령 RAG 한 바퀴 (통째로 넣으면? → 인덱스 → 검색 → 답변)
uv run python tutorials/01_ds_core/01_build_rag.py

# 3단계: 임베딩 속도 옵션 비교 (장치, 길이순 배치, 스레드)
uv run python tutorials/03_optimize/01_embedding_speed.py

# DS용 CLI
uv run ragkit --help
```

### 앱 (2·4단계)

네 앱 모두 `ragkit.service.Searcher`로 같은 인덱스를 씁니다. 자세한 사용법은 각 앱의 README를 보세요.

```bash
# 검색 API (FastAPI, 답변은 SSE 스트리밍, 노트북 저장) → http://127.0.0.1:8000/docs
uv run --package ragkit-api ragkit-api

# NotebookLM형 웹앱 '법령 노트' (api를 띄운 뒤) → http://localhost:5173
cd apps/web && pnpm install && pnpm dev

# 최종 사용자용 검색 CLI
uv run --package ragkit-search ragkit-search search "야간 근로 수당" --law 근로기준법
uv run --package ragkit-search ragkit-search ask "주휴수당은 누가 받아?"

# MCP 서버 (Claude Code 등록 예시는 apps/mcp/README.md)
uv run --package ragkit-mcp ragkit-mcp
```

| 앱 | 설명 |
|---|---|
| [`apps/api`](apps/api) | `/api/search`, `/api/answer/stream`, `/api/notebooks/...` |
| [`apps/web`](apps/web/README.md) | 노트북 = 법령 묶음, 인용 달린 답, 조문 보기, 노트 |
| [`apps/search-cli`](apps/search-cli/README.md) | `search` · `ask` · `laws` · `show` |
| [`apps/mcp`](apps/mcp/README.md) | `search_laws` · `get_article` · `list_laws` · `ask` |

## 저장소 구조

```
src/ragkit/      # 1단계 라이브러리: data · embeddings · retrieval · rag · training · evaluation · CLI
apps/            # 2~4단계: ragkit을 쓰는 앱 (uv workspace 멤버)
experiments/     # 실험 설정(config.yaml)과 결과
tutorials/       # 단계별 해설 (01_ds_core ~ 04_product, _legacy는 이전 구성 자료)
scripts/         # 일회성 데이터·모델 준비
docs/            # 강의 계획, 설계 문서, ADR
```

## 문서

- [강의 계획](docs/PLAN.md)
- [ragkit 설계](docs/superpowers/specs/2026-09-28-ragkit-restructure-design.md)
- [활용 앱 설계](docs/superpowers/specs/2026-09-29-apps-design.md) (api · search-cli · mcp · web)
- [아키텍처](docs/ARCHITECTURE.md) · [ADR](docs/adr/)
