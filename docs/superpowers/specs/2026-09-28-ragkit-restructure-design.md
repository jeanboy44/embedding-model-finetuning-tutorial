# 강의 재구성 + ragkit 패키지 설계 (2026-09-28)

브랜치: `feat/ragkit-phase01` (워크트리 `.claude/worktrees/ragkit`)
관련 문서: `docs/handoff-law-question-gen.md`(코퍼스·질문 생성), `feat/training-scripts`의 `2026-09-28-embedding-training-design.md`(학습·평가)

## 1. 강의 스토리라인 (확정)

1. **통째로 넣을 수 있는 문서는 그냥 LLM에 넘긴다.** 이 강의는 넣을 수 없는 경우(법령 코퍼스 약 700만 자)를 다룬다.
2. 넣을 수 없을 때의 선택지:
   - ① 에이전트형 탐색(파일 서치, 색인. Claude Code가 코드를 읽는 방식): 목적에 따라 품질이 더 좋을 수 있으나 **쿼리당 LLM 호출이 많아 비싸다**.
   - ② 임베딩 RAG: 쿼리당 비용이 싸다. 이 강의는 ②를 다룬다.
   - (한 줄 언급) RAG의 검색기는 키워드, SQL, 그래프 등 무엇이든 될 수 있지만, 여기서는 가장 흔한 임베딩 기반을 다룬다.
3. **RAG의 대가:** 검색이 놓치면 답도 틀린다. 검색 recall이 답변 정확도의 상한이다.
4. **주객전도:** 쿼리 확장, HyDE 같은 보완책으로 쿼리당 LLM 호출이 다시 늘어나 결국 ①처럼 비싸진다. (리랭커, 하이브리드 검색은 표준 구성이므로 "땜질" 사례에 넣지 않는다.)
5. **파인튜닝:** 쿼리마다 내던 비용을 학습 1회 비용으로 옮긴다.
6. **학습 데이터 생성:** LLM을 쓰지만 이것도 1회 비용이다. 손익분기 = 데이터 생성 비용 ÷ 쿼리 확장 1회 비용.
   - 운영: 비용이 적은 LLM API를 호출하는 스크립트로 구현하는 것이 좋다.
   - 개발: Claude Code / Antigravity 안에서 SKILL.md로 생성해 보는 것도 방법이다.
   - 이 강의: skill(`law-question-gen`)로 약식 진행한다.
7. **만능 아님:** 추론, 수치 필터, 코퍼스에 없는 지식은 파인튜닝으로 고쳐지지 않는다. 학습·재임베딩이라는 운영 트레이드오프도 있다. → Phase 4(에이전트)로 연결.

## 2. Phase 구성과 시간 (8시간 = 480분)

| Phase | 폴더 | 내용 | 시간 |
|---|---|---|---|
| 1 | `tutorials/phase01/` | 간단한 RAG 구축 + 순수 LLM / RAG / RAG+쿼리 확장 비교 | 45분 |
| 2 | `tutorials/phase02/` | 학습 데이터 생성(skill) → 파인튜닝(전체 / LoRA) → 평가 → ONNX 변환 | 165분 |
| 3 | `tutorials/phase03/` | CLI & MCP (기존 phase2_cli_mcp) | 90분 |
| 4 | `tutorials/phase04/` | RAG 에이전트 (기존 phase3_agent) | 100분 |
| 5 | `tutorials/phase05/` | 모니터링 (기존 phase4_monitoring) | 80분 |

- 각 Phase 폴더 안의 파일은 `01_...py`, `02_...py` 순서로 둔다.
- 튜토리얼은 **ragkit을 소비**한다. 튜토리얼에 로직을 다시 구현하지 않는다.

## 3. 저장소 구조

```
src/ragkit/            # 프로젝트 본체. 설치 가능한 패키지 (uv_build)
  config.py
  models/              # 임베딩 백엔드(onnx 기본, torch 선택), LLM 클라이언트
  embeddings/          # embed(texts) -> np.ndarray, 앞 문구(query:/passage:)
  retrieval/           # 인덱스 생성·저장·로드, 검색 (numpy 내적)
  rag/                 # 검색 → 답변 생성, 쿼리 확장
  data/                # 코퍼스·질문 로드, relevance_key(parent_id 우선)
  training/            # [train] 트리플 변환, 분할, 학습(sentence-transformers, LoRA), ONNX 변환
  evaluation/          # [train] Recall@k, MRR, nDCG
  monitoring/          # 로깅, 운영 메트릭 (core)
  mcp/
  cli/                 # ragkit index | search | ask | train | evaluate | export-onnx
experiments/           # 코드 없음. 실험별 config.yaml + 결과
  exp_000_pure_llm/ exp_001_base_rag/ exp_002_finetuned/ exp_003_query_expansion/ exp_004_lora/
scripts/               # 일회성 데이터 준비 (prepare_law_data, 모델 다운로드 등)
tutorials/phase01..05/ # 해설 + ragkit 호출
tests/
```

- 기존 `src/`(패키지 이름이 `src`)는 `src/ragkit/`로 옮기고 import를 `ragkit.`으로 바꾼다.
- 기존 `experiments/*/run.py`, `base_experiment.py`, `evaluate.py`(영어 5문장 예제)는 제거한다. 실험 실행은 ragkit CLI가 맡는다.
- 옛 `src/slm_finetuning_example.egg-info` 제거.

## 4. 의존성 (배포 크기 최소화)

| 구분 | 설치 | 내용 |
|---|---|---|
| core (배포 기본) | `ragkit` | onnxruntime, tokenizers, numpy, pydantic-settings, google-genai, cyclopts, loguru |
| extra `[torch]` | `ragkit[torch]` | torch, transformers |
| extra `[train]` | `ragkit[train]` | `[torch]` + sentence-transformers, peft, datasets, pyyaml, onnx |
| group `dev` | 저장소 전용 | pytest, ruff, mypy |
| group `tutorial` | 저장소 전용 | scikit-learn(부록 PCA), gdown, huggingface-hub 등 |

- 제거: langchain, langgraph, fastapi, uvicorn, polars (코드에서 쓰지 않음). scikit-learn은 core에서 제거(cosine_similarity → numpy 내적. 임베딩은 이미 L2 정규화).
- `[train]`/`[torch]` 모듈의 무거운 import는 함수 안에서 한다. extra 없이 호출하면 설치 명령을 안내하는 오류를 낸다. core만 설치해도 `import ragkit`은 동작해야 한다.
- 강의: `uv sync --all-extras`. 배포: `uv sync --no-dev --no-default-groups`(core만).

## 5. 임베딩 백엔드

- 설정 `embedding_backend: onnx | torch` (기본 `onnx`).
- 공통 인터페이스: `create_embedding_fn(model_dir, backend=...) -> Callable[[list[str]], np.ndarray]` (L2 정규화된 mean pooling).
- onnx: `tokenizers`(모델 폴더의 `tokenizer.json`) + `onnxruntime`(`<모델 폴더>/onnx/model.onnx`) + numpy pooling.
- torch: 기존 transformers 경로. 체크포인트가 폴더(ST 형식)면 `from_pretrained(폴더)`.
- `ragkit export-onnx <모델 폴더>` ([train]): base·파인튜닝 모델 모두 이 명령으로 변환한다. 흐름: 학습(torch) → 변환 → 배포(onnx).
- 검증: 같은 문장에서 torch와 onnx 출력의 코사인 유사도 ≥ 0.999 (테스트).
- ONNX 모델이 없으면 `export-onnx` 실행을 안내하고 종료한다.

## 6. Phase 1 상세 (이번 브랜치 범위)

코퍼스: `data/processed/law_docs.json`의 **youth 테마**(약 2,400청크).

| 파일 | 내용 |
|---|---|
| `01_build_rag.py` | 코퍼스 로드 → "통째로 넣으면?" 토큰 계산 → 인덱스 생성(`data/processed/index/youth_base/`, 캐시) → 검색 → 답변 생성 |
| `02_pure_llm.py` | 검색 없이 LLM만. 최근 개정 조문 질문에서 틀리는 장면 |
| `03_base_rag.py` | RAG (base 임베딩). Recall@5, MRR |
| `04_query_expansion.py` | RAG + LLM 쿼리 확장. Recall, 쿼리당 LLM 호출·토큰·지연 |
| `05_compare.py` | 02~04 결과 표 + 주객전도 해설 → Phase 2 연결 |
| `appendix_exploration.py` | 기존 01_exploration (선택 실습) |

- 비교 표: 방식 / Recall@5 / MRR / 쿼리당 LLM 호출 / 입력 토큰 / 지연. 답변 정확도는 몇 개 질문의 답을 나란히 보여 주고 수강생이 판단한다(LLM judge 없음).
- API 키가 없으면 LLM 단계는 건너뛰고 검색 지표만 출력한다.
- 평가 쿼리: `data/eval/youth_eval.jsonl` (커밋). 20개 초안, **LLM 초안이므로 강사 검수 필요**라고 파일 머리 설명(README 또는 별도 md)에 명시한다. 형식은 skill 출력과 같다(`query, positive_id, query_type, answer`).
- 결과 JSON: `experiments/<exp>/results/`(gitignore).

## 7. 미정 (사용자 결정 필요)

- **학습/평가 분할 방식:** 법령 단위(`feat/training-scripts` 설계) vs 같은 코퍼스 안에서 쿼리 단위 + 사람이 쓴 평가셋(에이전트 토론 권장). Phase 2 설계 때 결정.
- Phase 2 평가 범위: youth 테마 vs 전체 코퍼스.

## 8. 작업 순서

1. ragkit 골격: `src/ragkit/` 이동, import 치환, pyproject(uv_build, 의존성, extras, groups, `[project.scripts]`), sklearn 제거, 테스트 통과.
2. 임베딩 백엔드 onnx/torch + export-onnx.
3. 튜토리얼 폴더 이동(`phase01`~`phase05`), experiments 정리, README.
4. Phase 1 구현 + 평가셋 초안.
5. main 병합 시: `scripts/prepare_law_data.py` 등 `from src.` import를 `ragkit.`으로 수정(다른 세션 작업 커밋 후).
