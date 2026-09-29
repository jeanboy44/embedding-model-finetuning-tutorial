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
  web/                 # 4단계. React 검색 화면 (Node 프로젝트, workspace 멤버 아님, apps/api 호출)
experiments/           # 코드 없음. 실험별 config.yaml + 결과
tutorials/             # 단계별 해설. ragkit과 apps를 호출만 한다
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
- 베이스 모델 비교: `intfloat/multilingual-e5-small` 등 후보를 같은 평가셋으로 비교 (`docs/adr/001-model-selection.md`)
- 데이터 준비: legalize-kr 법령 코퍼스 (`scripts/prepare_law_data.py`, 긴 조문은 항·호 단위로 나눠 약 2.6만 문서), Claude 스킬로 질문·정답 조문·hard negative 생성 (`.claude/skills/law-question-gen/`), 질문 공유는 `scripts/law_questions_drive.py`
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

- `ragkit export-onnx`로 PyTorch → ONNX 변환, 양자화(동적 INT8 등)
- `apps/bench`: 원본 / ONNX / 양자화 모델의 **지연 시간·메모리·모델 크기·설치 크기·검색 정확도(Recall@k)** 비교표
- `apps/api`의 백엔드를 ONNX로 바꾸고 torch 없는 배포 이미지 크기를 비교한다
- 메시지: "정확도를 얼마나 잃고 얼마나 빨라지는가"를 DS가 직접 재고 판단한다

### 4단계. AI가 오면서: CLI 도구, MCP, 프론트엔드 (apps/search-cli, apps/mcp, apps/web)

> MCP를 4단계에 두는 것은 제안이다(apps에 넣는 것은 확정).

AI 도구 덕분에 DS가 직접 손대기 쉬워진 영역 ②. 모델을 "제품"으로 전달한다.

- `apps/search-cli`: 동료가 설치해서 바로 쓰는 검색 도구 (cyclopts, `uvx`로 실행, ragkit core + ONNX만)
- `apps/mcp`: MCP 서버로 Claude 같은 에이전트가 조문을 검색하게 한다
- `apps/web`: 질문을 넣으면 관련 조문을 보여 주는 React 검색 화면 (2단계 API 호출)
- 메시지: DS 한 사람이 모델부터 사용자 화면까지 끝까지 만든다

## 기존 자료와의 관계

- `README.md`의 Phase 1~4(모델 개발 / CLI & MCP / RAG 에이전트 / 모니터링)와 `tutorials/phase1_model_dev/` 등은 이 계획 이전의 구성이다.
- 옮길 곳: 모델 개발 → 1단계, CLI & MCP → 4단계(`apps/search-cli`, `apps/mcp`). RAG 에이전트·모니터링은 미정(아래).
- README 개편은 아직 하지 않았다.

## 미정 사항

- 단계별 시간 배분 (전체 8시간 = 480분). 초안(제안, 미확정): 1단계 210분(도입 45 + 데이터·학습·평가 165) / 2단계 60분 / 3단계 90분 / 4단계 120분(CLI 40 · MCP 30 · React 50)
- RAG 에이전트(`ragkit.rag`)와 모니터링: 1단계 도입에서만 쓸지, 부록으로 둘지, 뺄지
- 2단계 API 세부(배포 방식: Docker 여부), 4단계 프론트엔드 스택(빌드 도구, UI 라이브러리)
