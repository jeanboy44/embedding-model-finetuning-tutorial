# 임베딩 파인튜닝 학습·평가 설계 (2026-09-28, ragkit 구조 반영)

브랜치: `feat/training-scripts` (워크트리 `.claude/worktrees/training-scripts`)
상위 설계: `feat/ragkit-phase01`의 `docs/superpowers/specs/2026-09-28-ragkit-restructure-design.md`
이 문서는 ragkit 설계의 `ragkit.data` / `ragkit.training` / `ragkit.evaluation` 부분을 구체화한다.

## 목표
법령 질문 데이터(`data/questions/*.jsonl`)가 이미 있다고 가정하고, 그 이후 단계를 만든다.
분할 → 학습(전체 학습 / LoRA) → 전체 코퍼스 대상 검색 평가 → base / 전체 학습 / LoRA 비교.
강의(Phase 2)에서 학생이 CLI로 돌리고, 튜토리얼이 이 함수들을 호출한다.

## 결정 사항
| 항목 | 결정 | 이유 |
|---|---|---|
| 학습 라이브러리 | sentence-transformers | 실무 표준. 로컬 `models/multilingual-e5-small`이 이미 ST 형식 |
| 분할 | **법령 단위** (테마마다 법령 ~20% test, ~10% dev) | 학습 때 본 적 없는 법령에서도 좋아지는지 측정. 조문 누수 없음. (ragkit 설계 7절 "미정" 항목을 이것으로 확정) |
| 실험 | 전체 학습(`exp_002_finetuned`)과 LoRA(`exp_004_lora`)를 별도 실험으로 | 이 크기 모델에서 무엇이 나은지 데이터로 비교 |
| 평가 대상 | 전체 코퍼스(약 1.2만 조문) | 실제 검색 상황과 같게 |
| 임베딩 백엔드 | 평가는 `embed_fn: Callable[[list[str]], np.ndarray]`만 받는다 | onnx / torch 백엔드와 무관하게 동작 |

## 입력 형식 (가정)
- 코퍼스 `data/processed/law_docs.json`: `[{id, title, text, category, theme, law_name, ..., parent_id?}]`
  - `parent_id`(긴 조문 분할)는 있으면 쓰고, 없으면 `id`로 대신한다.
- 질문 `data/questions/<법령>__pNN.jsonl`, 한 줄에 하나:
  `{"query", "positive_id", "hard_negative_ids": [1~3개], "query_type": "situation|question|keyword", "answer"}`
  - 질문의 법령·테마는 `positive_id`의 코퍼스 레코드(`category`, `theme`)에서 얻는다.

## 모듈 (`src/ragkit/`)

### `ragkit/data/` (core, 무거운 의존성 없음)
- `load_corpus(path) -> list[dict]`, `load_questions(paths_or_dir) -> list[dict]`
- `relevance_key(doc) -> str`: `parent_id`가 있으면 그것, 없으면 `id`
- `doc_text(doc) -> str`: `title + "\n" + text` (인덱스·학습·평가가 같은 문서 텍스트를 쓰도록 한 곳에서 정의)

### `ragkit/training/split.py` (core 의존성만)
- `split_by_law(questions, corpus, *, test_ratio=0.2, dev_ratio=0.1, seed=42) -> dict[str, list[dict]]`
  - 테마마다 법령을 seed로 섞고, 질문 수 누적이 비율에 닿을 때까지 test → dev에 배정, 나머지 train.
  - 테마에 법령이 2개 이하이면 test에만 1개 뺀다(train이 비지 않게).
  - 분할 사이에 법령이 겹치지 않음을 검사한다.
- `write_splits(splits, out_dir)`: `train/dev/test.jsonl` + `split_meta.json`(seed, 비율, 분할별 법령 목록·질문 수)

### `ragkit/training/triplets.py` (core 의존성만)
- `to_triplets(questions, corpus, *, num_negatives=1, query_prefix, passage_prefix) -> tuple[list[dict], int]`
  - `{anchor, positive, negative_1..n}` 목록과 건너뛴 질문 수. 코퍼스에 없는 id가 있는 질문은 건너뛴다.
  - hard negative가 `num_negatives`보다 적은 질문은 있는 만큼 채우고, 0개면 `(anchor, positive)`만 쓰는 대신
    전체 목록의 형식을 맞추기 위해 건너뛴다(수를 보고).

### `ragkit/training/train.py` ([train] extra, import는 함수 안)
- `train(config: TrainConfig, train_questions, dev_questions, corpus, out_dir) -> dict`
  - `SentenceTransformer(모델 폴더)`, `MultipleNegativesRankingLoss`, `BatchSamplers.NO_DUPLICATES`
  - 기본값: batch 32, epoch 1, lr 2e-5, warmup_ratio 0.1, max_seq_length 512, seed 42
  - dev: `InformationRetrievalEvaluator`(dev 질문 × 전체 코퍼스)로 학습 전·후 점수
  - LoRA: `config.lora`가 있으면 `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=[query, key, value])`
    → `model.add_adapter(...)`. 저장은 병합한 전체 모델(ST 폴더). `save_adapter: true`면 어댑터도 따로 저장.
  - 장치 자동 선택(cuda → mps → cpu). `max_steps`, `limit`(train 질문 수 제한)로 강의용 짧은 실행.
  - `<out_dir>/train_meta.json`: 학습 시간, 전체/학습 대상 파라미터 수, 설정, dev 점수
- `TrainConfig`: pydantic 모델. 실험 `config.yaml`의 `training:` 블록을 읽는다.
- 저장 폴더는 ST 형식이므로 ragkit torch 백엔드(`from_pretrained(폴더)`)와 `ragkit export-onnx`가 그대로 읽는다.

### `ragkit/evaluation/` (core 의존성만: numpy)
- `retrieval_metrics(ranked_keys, positive_key, ks=(1, 5, 10)) -> dict`: 질문 하나의 Recall@k, MRR@10, nDCG@10
- `evaluate_retrieval(embed_fn, corpus, questions, *, ks=(1, 5, 10), query_prefix, passage_prefix, cache_path=None) -> dict`
  - 코퍼스 전체를 한 번 임베딩(캐시 `.npy` 선택), 질문마다 상위 후보를 뽑는다.
  - 정답 판정은 `relevance_key`. 같은 key의 조각이 여러 개면 순위에서 첫 등장만 센다.
  - 결과: 전체 평균, `query_type`별, `theme`별, 질문 수.

### CLI (`ragkit/cli/`에 하위 명령 추가)
- `ragkit split [--questions data/questions] [--corpus ...] [--out data/splits] [--seed 42]`
- `ragkit train --config experiments/exp_002_finetuned/config.yaml [--max-steps N] [--limit N]`
- `ragkit evaluate --model <모델 폴더> [--split test] [--out experiments/<exp>/results/]`
- 경로 기본값은 `ragkit.config` 설정 기준, 모두 인자로 바꿀 수 있다.

### 실험 설정 (코드 없음)
- `experiments/exp_002_finetuned/config.yaml`: `model`, `output_dir: models/finetuned/exp_002`, `training:` 블록
- `experiments/exp_004_lora/config.yaml`: 위와 같고 `training.lora:` 블록 추가
- base(`exp_001_base_rag`)와의 비교 표는 `ragkit evaluate` 결과 JSON을 읽어 Phase 2 튜토리얼에서 출력한다.

## 오류 처리
- 질문의 id가 코퍼스에 없음: 건너뛰고 경고(개수). `--strict`이면 중단.
- 질문 파일 없음, 분할 결과가 빔, 모델 폴더 없음: 먼저 실행할 명령을 안내하고 종료(코드 1).
- `[train]` extra 없이 `train` 호출: 설치 명령(`uv sync --extra train`)을 안내하는 오류.

## 테스트
- 가짜 코퍼스(3개 테마 × 법령 3~4개, 조문 약 40개, 일부 `parent_id` 조각) + 가짜 질문 fixture
- `test_data.py`: `relevance_key`, `doc_text`, 질문 로드(폴더/파일)
- `test_split.py`: 법령 누수 없음, 비율 근사, seed 재현성, 작은 테마
- `test_triplets.py`: 앞 문구, 없는 id 건너뛰기, negative 개수
- `test_evaluation.py`: 가짜 embed_fn으로 지표를 손계산 값과 비교, 조각 중복 처리
- `test_train_smoke.py` (`@pytest.mark.slow`, [train] 필요): 로컬 e5-small로 전체/LoRA 각각 2 step → 저장 → 다시 읽어 평가
- 마지막 검증: 실제 코퍼스 + 가짜 질문 약 200개로 split → train(짧게) → evaluate 전체 흐름

## 작업 순서와 ragkit 브랜치 의존
ragkit 골격(`src/ragkit/`, pyproject extras)이 아직 커밋되지 않았으므로 두 단계로 나눈다.

1. **골격 없이 가능한 부분 (지금):** `ragkit/data`, `training/split.py`, `training/triplets.py`, `evaluation/`과 테스트.
   - 이 모듈들은 서로만 import하고 `ragkit.config` 등 다른 ragkit 모듈에 의존하지 않는다(앞 문구는 인자).
   - 임시로 `src/ragkit/__init__.py`를 두고, 테스트는 `tests/conftest.py`에서 `src/`를 `sys.path`에 추가한다.
     rebase 때 골격 쪽 파일을 따르고 이 임시 장치는 지운다.
   - pyproject는 건드리지 않는다(충돌 방지).
2. **골격 커밋 후 rebase:** `training/train.py`([train] extra), CLI 하위 명령, 실험 config 2개, smoke 테스트, 전체 흐름 검증.

## 범위 밖
여러 GPU, 하이퍼파라미터 탐색, Hub 업로드, ONNX 변환(ragkit 설계 담당), 쿼리 확장, 질문 생성, Phase 2 튜토리얼 본문.
