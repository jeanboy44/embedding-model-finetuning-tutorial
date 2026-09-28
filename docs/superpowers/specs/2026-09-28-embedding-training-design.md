# 임베딩 파인튜닝 학습·평가 설계 (2026-09-28, ragkit 구조 반영)

브랜치: `feat/training-scripts` (워크트리 `.claude/worktrees/training-scripts`, `feat/ragkit-phase01` 4bbf396 위)
상위 문서: `docs/PLAN.md`(1단계 DS 본업), `docs/superpowers/specs/2026-09-28-ragkit-restructure-design.md`
이 문서는 ragkit의 `ragkit.data` / `ragkit.training` / `ragkit.evaluation`과 CLI `split` / `train` / `evaluate`를 정한다.

## 목표
법령 질문 데이터가 이미 있다고 가정하고, 그 이후 단계를 만든다.
분할 → 학습(전체 학습 / LoRA) → 전체 코퍼스 대상 검색 평가 → base / 전체 학습 / LoRA 비교.
강의 1단계에서 학생이 CLI로 돌리고, 튜토리얼과 apps가 이 함수들을 호출한다.

## 결정 사항 (사용자 확정)
| 항목 | 결정 | 이유 |
|---|---|---|
| 학습 라이브러리 | sentence-transformers (설치 버전 6.1) | 실무 표준. 로컬 `models/multilingual-e5-small`이 이미 ST 형식 |
| 분할 | **법령 단위** (테마마다 법령 ~20% test, ~10% dev) | 학습 때 본 적 없는 법령에서도 좋아지는지 측정. 조문 누수 없음 |
| 실험 | 전체 학습(`exp_002_finetuned`)과 LoRA(`exp_004_lora`)를 별도 실험으로 | 이 크기 모델에서 무엇이 나은지 데이터로 비교 |
| 평가 대상 | 전체 코퍼스 | 실제 검색 상황과 같게 |
| 백엔드 무관 | 평가는 `embed_fn: Callable[[list[str]], np.ndarray]`만 받는다 | onnx / torch 백엔드 어느 쪽이든 동작. Phase 1 튜토리얼도 이 함수를 쓴다 |

## 입력 데이터 (실제 확인한 형식)
- 코퍼스 `data/processed/law_docs.json`: 문서 25,806개(긴 조문을 항·호 단위로 나눔). 모든 레코드에 `parent_id`, `paragraph`가 있다.
  - 예: `id = 전기사업법_법률_제2조_제16~21호`, `parent_id = 전기사업법_법률_제2조`. 나누지 않은 조문은 `parent_id == id`.
  - `category`(법령 폴더 이름), `theme`로 법령·테마를 안다.
- 질문 JSONL(한 줄에 하나). 스킬 출력(`data/questions/<법령>__pNN.jsonl`)과 테스트 모음(`data/questions_test/questions.jsonl`, `feat/law-question-gen`)이 같은 핵심 필드를 쓴다:
  - `query`, `positive_id`, `hard_negative_ids`(1~5개), `query_type`(`situation|question|keyword`, 없으면 null), `answer`(없으면 null)
  - 선택 필드: `related_ids`(정답을 부분적으로 담은 문서. negative로 쓰면 안 됨), `corpus`(id가 속한 코퍼스 파일 이름), `source`
- 테스트 모음 중 현재 코퍼스와 맞는 권장 데이터: `source.iteration == 2 and source.config == "with_skill"` → 259개, 법령 3개(도로교통법 169, 최저임금법 46, 전기공사공제조합법 44), 모든 id가 코퍼스에 있음.

## 모듈 (`src/ragkit/`)

### `ragkit/data/` (core 의존성만)
- `load_corpus(path) -> list[dict]`
- `load_questions(path) -> list[dict]`: JSONL 파일 하나, 또는 폴더(안의 `*.jsonl` 전부, 이름순)
- `relevance_key(doc) -> str`: `parent_id`가 있으면 그것, 없으면 `id`
- `doc_text(doc) -> str`: `title + "\n" + text`. 인덱스·학습·평가가 같은 문서 텍스트를 쓰도록 한 곳에서 정의
- `filter_questions(questions, corpus_by_id) -> tuple[list[dict], dict]`
  - `positive_id`가 코퍼스에 없는 질문은 뺀다(옛 코퍼스 id를 쓰는 질문이 여기서 걸러진다).
  - `hard_negative_ids`에서 코퍼스에 없는 id, `related_ids`에 있는 id, **positive와 같은 조(`relevance_key`)의 조각**은 지운다(가짜 negative 방지).
  - 두 번째 반환값은 건너뛴 수 통계(`missing_positive`, `dropped_negatives`).
  - `corpus` 필드로 거르지 않고 id 존재로 거른다(필드가 없는 스킬 출력에도 똑같이 동작).

### `ragkit/training/split.py` (core 의존성만)
- `split_by_law(questions, corpus_by_id, *, test_ratio=0.2, dev_ratio=0.1, seed=42) -> dict[str, list[dict]]`
  - 질문의 법령·테마는 positive 문서의 `category`, `theme`.
  - 테마마다 법령을 seed로 섞고, test 질문 수가 `test_ratio`에 닿을 때까지 법령을 test에 배정, 이어서 dev, 나머지 train.
  - 법령이 3개 미만인 테마들은 한 그룹으로 합쳐 같은 규칙을 적용한다.
  - 그룹마다 train에 법령이 최소 1개 남게 한다. 그래서 dev나 test가 빌 수 있고, 그러면 경고한다.
  - 분할 사이에 법령이 겹치지 않음을 검사한다.
- `write_splits(splits, out_dir, meta)`: `train/dev/test.jsonl` + `split_meta.json`(seed, 비율, 분할별 법령 목록·질문 수)

### `ragkit/training/triplets.py` (core 의존성만)
- `to_examples(questions, corpus_by_id, *, num_negatives=1, query_prefix, passage_prefix) -> list[dict]`
  - `{anchor, positive, negative_1..n}`. 텍스트는 앞 문구 + `doc_text`.
  - hard negative가 `num_negatives`보다 적은 질문은 있는 negative를 반복해 채운다(모든 행의 열 수가 같아야 한다). 0개면 건너뛴다.
  - 입력은 `filter_questions`를 거친 질문이라고 가정한다.

### `ragkit/training/train.py` ([train] extra, 무거운 import는 함수 안)
- `TrainConfig`(pydantic): `model`, `output_dir`, `batch_size=32`, `epochs=1`, `lr=2e-5`, `warmup_ratio=0.1`,
  `max_seq_length=512`, `num_negatives=1`, `seed=42`, `max_steps=None`, `limit=None`, `lora: LoraSettings | None`
  - `LoraSettings`: `r=16`, `alpha=32`, `dropout=0.1`, `target_modules=["query", "key", "value"]`, `save_adapter=False`
  - `load_train_config(path)`: 실험 `config.yaml`을 읽는다.
- `train(config, train_questions, dev_questions, corpus, *, query_prefix, passage_prefix) -> dict`
  - `SentenceTransformer(모델 폴더)`, `MultipleNegativesRankingLoss`, `BatchSamplers.NO_DUPLICATES`, `SentenceTransformerTrainer`
  - dev가 있으면 `InformationRetrievalEvaluator`(dev 질문 × 전체 코퍼스, 정답은 positive id)로 학습 전·후 점수
  - LoRA: `peft.get_peft_model(model[0].auto_model, LoraConfig(...))`로 감싸 학습 → `merge_and_unload()`로 합쳐
    ST 폴더로 저장. `save_adapter`면 합치기 전에 어댑터를 `<output_dir>/adapter/`에 따로 저장
  - 장치 자동 선택(cuda → mps → cpu).
  - `<output_dir>/train_meta.json`: 학습 시간, 전체/학습 대상 파라미터 수, 설정, 학습 질문 수, dev 점수
- 저장 폴더는 ST 형식이라 ragkit torch 백엔드(`from_pretrained(폴더)`)와 `ragkit export-onnx`가 그대로 읽는다.

### `ragkit/evaluation/` (core 의존성만: numpy)
- `rank_of(ranked_ids, positive_id, *, ignore=()) -> int | None`: `ignore`(related_ids)를 순위에서 빼고 정답의 순위(1부터)
- `question_metrics(rank, ks) -> dict`: Recall@k(=hit@k, 정답이 하나), MRR@10, nDCG@10
- `evaluate_retrieval(embed_fn, corpus, questions, *, ks=(1, 5, 10), query_prefix="query: ", passage_prefix="passage: ", corpus_embeddings=None) -> dict`
  - 코퍼스 전체를 한 번 임베딩(`corpus_embeddings`를 주면 재사용), 내적으로 상위 후보를 뽑는다.
  - 판정 두 가지를 함께 낸다:
    - `doc`: 정답 문서 id와 정확히 일치(기본 지표)
    - `article`: 같은 조(`relevance_key`)의 조각이면 정답. 같은 조의 조각이 여러 개 나오면 첫 등장만 센다
  - `related_ids`는 두 판정 모두에서 순위에서 뺀다(오답으로도 정답으로도 치지 않음).
  - 결과: `{"n", "doc": {...}, "article": {...}, "by_query_type": {...}, "by_theme": {...}}`
- 코퍼스 임베딩 캐시는 CLI에서만 한다(`experiments/<exp>/results/corpus_emb.npy`, 모델 폴더·코퍼스 파일 수정 시각이 바뀌면 다시 계산).

### CLI (`ragkit/cli/cli_tool.py`의 `app`에 하위 명령 추가)
- `ragkit split --questions <파일|폴더> [--corpus ...] [--out data/splits] [--seed 42] [--test-ratio 0.2] [--dev-ratio 0.1]`
- `ragkit train --config experiments/exp_002_finetuned/config.yaml [--splits data/splits] [--corpus ...] [--max-steps N] [--limit N]`
- `ragkit evaluate --model <모델 폴더|이름> [--split test] [--splits data/splits] [--corpus ...] [--out <결과 JSON>]`
  - 모델은 이번 브랜치에서는 `sentence_transformers.SentenceTransformer`로 불러 `embed_fn`을 만든다([train] extra).
    ragkit 세션의 onnx/torch 백엔드가 들어오면 `ragkit.embeddings.create_embedding_fn`으로 바꾼다.
- 경로 기본값: `get_settings().data_dir` 기준. 모두 인자로 바꿀 수 있다(워크트리처럼 `data/`가 없는 곳에서 `--corpus`로 지정).

### 실험 설정 (코드 없음)
- `experiments/exp_002_finetuned/config.yaml`: `name`, `training:` 블록(`TrainConfig`), `output_dir: models/finetuned/exp_002`
- `experiments/exp_004_lora/config.yaml`: 위와 같고 `training.lora:` 블록, `output_dir: models/finetuned/exp_004`
- 기존 `experiments/exp_00X_*/run.py` 정리는 ragkit 세션 담당이라 손대지 않는다.

## 오류 처리
- 질문 필터 결과: 뺀 질문·negative 수를 출력. `--strict`이면 뺀 질문이 하나라도 있을 때 중단.
- 질문 파일 없음, 분할 결과가 빔, train이 빔, 모델 폴더 없음: 먼저 실행할 명령을 안내하고 종료(코드 1).
- `[train]` extra 없이 `train`/`evaluate`: `uv sync --extra train`을 안내하는 오류.

## 테스트
- 가짜 코퍼스(테마 3개: 법령 4개 테마 1개 + 법령 1~2개 테마 2개, 조각 있는 조 포함) + 가짜 질문 fixture (`tests/conftest.py`)
- `test_data.py`: `relevance_key`, `doc_text`, 질문 로드(파일/폴더), 필터(없는 positive, 같은 조 negative, related 제거)
- `test_split.py`: 법령 누수 없음, 비율 근사, seed 재현성, 작은 테마 합치기, train 최소 1개
- `test_triplets.py`: 앞 문구, negative 채우기, negative 0개 건너뛰기
- `test_evaluation.py`: 가짜 embed_fn으로 지표를 손계산 값과 비교, `doc`/`article` 판정, related 무시
- `test_train_smoke.py` (`@pytest.mark.slow`): 로컬 e5-small로 전체/LoRA 각각 2 step → 저장 → 다시 불러 평가 → LoRA 병합본에 peft 흔적이 없는지
- 마지막 검증: 실제 코퍼스(25,806) + 권장 테스트 질문 259개로 split → train(짧게, 전체/LoRA) → evaluate(base/전체/LoRA) 전체 흐름.
  질문이 법령 3개뿐이라 수치는 참고용이고, 흐름이 끝까지 도는지를 본다.

## 알려진 제약
- `ragkit.config.Settings.project_root`가 4bbf396에서 `src/`를 가리킨다(ragkit 세션에 알림). 수정 전에는 경로 인자를 직접 준다.
- MNRL의 in-batch negative에서 `related_ids`는 뺄 수 없다(같은 배치에 들어올 확률이 낮아 무시).

## 범위 밖
여러 GPU, 하이퍼파라미터 탐색, Hub 업로드, ONNX 변환·양자화(ragkit 세션), 임베딩 백엔드(ragkit 세션), 쿼리 확장, 질문 생성, 튜토리얼 본문.
