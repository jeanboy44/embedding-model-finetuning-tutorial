# 임베딩 파인튜닝 학습·평가 스크립트 설계 (2026-09-28)

## 목표
법령 질문 데이터(`data/questions/*.jsonl`)가 이미 있다고 가정하고, 그 이후 단계를 만든다.
분할 → 학습(전체 학습 / LoRA) → 전체 코퍼스 대상 검색 평가 → 세 모델 비교.
8시간 강의에서 학생이 직접 돌리고 읽을 코드이므로 짧고 읽기 쉬워야 한다.

## 결정 사항
| 항목 | 결정 | 이유 |
|---|---|---|
| 학습 라이브러리 | sentence-transformers | 실무 표준. 로컬 `models/multilingual-e5-small`이 이미 ST 형식(`modules.json`, `1_Pooling/`) |
| 분할 | 법령 단위 (테마마다 법령 ~20% test, ~10% dev) | 학습 때 본 적 없는 법령에서도 좋아지는지 측정. 조문 누수 없음 |
| 실험 | 전체 학습과 LoRA를 별도 실험으로 | 강의에서 "이 크기 모델에서 무엇이 나은가"를 데이터로 비교 |
| 평가 대상 | 항상 전체 코퍼스(약 1.2만 조문) | 실제 검색 상황과 같게 |

## 입력 형식 (가정)
- 코퍼스 `data/processed/law_docs.json`: `[{id, title, text, category, theme, law_name, ..., parent_id?}]`
  - `parent_id`는 다른 세션에서 추가 중인 필드(긴 조문 분할). 있으면 쓰고 없으면 `id`로 대신한다.
- 질문 `data/questions/<법령>__pNN.jsonl`, 한 줄에 하나:
  `{"query", "positive_id", "hard_negative_ids": [1~3개], "query_type": "situation|question|keyword", "answer"}`
  - 파일 이름의 `<법령>`은 코퍼스의 `category`(법령 폴더 이름)와 같다.

## 구성

```
data/questions/*.jsonl ──┐
data/processed/law_docs.json ──┤
                         ├─> scripts/split_questions.py
                         │     → data/splits/{train,dev,test}.jsonl, data/splits/split_meta.json
                         ├─> scripts/train_embedding.py --config experiments/exp_00X/config.yaml
                         │     → models/finetuned/<실험 이름>/  (ST 형식 폴더, LoRA는 병합 후 저장)
                         └─> scripts/evaluate_retrieval.py --model <이름|폴더> [--split test]
                               → experiments/results/<모델>_retrieval.json
```

### `src/training/` (로직은 여기, 스크립트는 인자 처리만)
- `data.py`
  - `load_corpus(path) -> list[dict]`, `load_questions(paths) -> list[dict]`
  - `relevance_key(doc) -> str`: `parent_id`가 있으면 그것, 없으면 `id`
  - `to_triplets(questions, corpus, *, num_negatives=1) -> list[dict]`: `{anchor, positive, negative_1..n}`,
    각 텍스트에 `query: ` / `passage: ` 앞 문구를 붙인다(`src.embeddings.format_queries/format_passages`).
    문서 텍스트는 `title + "\n" + text`. 코퍼스에 없는 id를 가진 질문은 건너뛰고 수를 돌려준다.
- `split.py`
  - `split_by_law(questions, law_of, *, test_ratio=0.2, dev_ratio=0.1, seed=42) -> dict[str, list]`
  - 테마마다 법령을 seed로 섞고, 질문 수 누적이 비율에 닿을 때까지 test → dev에 배정, 나머지 train.
    테마에 법령이 2개 이하이면 test에 1개만 뺀다(train이 비지 않게).
  - 분할 사이에 법령이 겹치지 않음을 assert.
- `evaluate.py`
  - `evaluate_retrieval(embed_fn, corpus, questions, *, ks=(1, 5, 10)) -> dict`
    - 코퍼스 전체를 한 번 임베딩, 질문마다 상위 max(k)를 뽑는다.
    - 정답 판정은 `relevance_key`로 비교(분할 조각 중 하나라도 맞으면 정답). 순위는 같은 key의 첫 등장만 센다.
    - 지표: Recall@1/5/10, MRR@10, nDCG@10. 전체, `query_type`별, `theme`별로 계산.
  - 코퍼스 임베딩 캐시: `experiments/results/cache/<모델 이름>_<코퍼스 해시>.npy`

### 실험
- `exp_001_base_embedding`: 학습 없음. `evaluate_retrieval`로 test 지표를 낸다.
- `exp_002_finetuned_embedding`: 전체 학습. `config.yaml`에 `training:` 블록, `checkpoint_path: models/finetuned/exp_002`.
- `exp_004_lora_embedding` (새로 만듦): 같은 구조 + `training.lora: {r: 16, alpha: 32, dropout: 0.1, target_modules: [query, key, value]}`.
- `exp_003`은 이번 범위 밖(손대지 않음).
- `experiments/evaluate.py`와 `tutorials/phase1_model_dev/run_experiments.py`에 exp_004를 추가해 세 모델의 test 지표와 학습 시간을 한 표로 출력한다.
- 영어 예제 5문장(`TEST_DOCS`)은 exp_001/002에서 제거하고, 데이터가 없으면 준비 순서를 안내한다.

### 학습 (`scripts/train_embedding.py`)
- `SentenceTransformer(resolve_model_source(model_name))`
- 손실: `MultipleNegativesRankingLoss` (in-batch negative + hard negative)
- `BatchSamplers.NO_DUPLICATES` (같은 조문이 한 배치에 두 번 → 가짜 negative 방지)
- 기본값: batch 32, epoch 1, lr 2e-5, warmup_ratio 0.1, max_seq_length 512, seed 42
- dev: `InformationRetrievalEvaluator`(dev 질문 × 전체 코퍼스)로 학습 전·후 점수
- LoRA: `config.training.lora`가 있으면 `peft.LoraConfig`로 `model.add_adapter(...)`.
  저장은 병합한 전체 모델(`merge_and_unload`) → exp_002와 같은 방식으로 읽힘. 옵션으로 어댑터만 따로 저장.
- 장치 자동 선택(cuda → mps → cpu). `--max-steps`, `--limit`(train 질문 수 제한)로 강의용 짧은 실행.
- 학습 시간·파라미터 수(전체/학습 대상)를 `<출력 폴더>/train_meta.json`에 남긴다.

### 모델 로더 변경
- `src/models/embedding_loader.load_embedding_model`: `checkpoint_path`가 폴더이면 `AutoModel.from_pretrained(폴더)`,
  파일이면 기존 state_dict 방식. 토크나이저도 폴더에 있으면 폴더에서 읽는다.
  (ST 폴더 루트에 HF 모델 `config.json`과 가중치가 있으므로 기존 mean pooling 경로가 그대로 동작)

### CLI
- cyclopts, 기존 `scripts/*.py`와 같은 형식(`run(main)`).
- 경로 기본값은 `get_settings().data_dir` 기준, 모두 인자로 바꿀 수 있다(워크트리처럼 `data/`가 없는 곳에서 `--corpus`로 지정).

## 오류 처리
- 질문의 id가 코퍼스에 없음: 건너뛰고 경고(개수). `--strict`이면 중단.
- 질문 파일 없음, 분할 결과가 빔, 체크포인트 없음: 먼저 실행할 명령을 안내하고 종료(코드 1).

## 의존성
- `uv add sentence-transformers peft` (`datasets`는 ST 학습에 필요하면 함께)

## 테스트
- 가짜 코퍼스(3개 테마 × 법령 3~4개, 조문 약 40개) + 가짜 질문 fixture (`tests/conftest.py`)
- `test_training_data.py`: 트리플 변환, 앞 문구, 없는 id 건너뛰기, `relevance_key`
- `test_split.py`: 법령 누수 없음, 비율 근사, seed 재현성, 작은 테마
- `test_evaluate.py`: 가짜 embed_fn으로 지표를 손계산 값과 비교, `parent_id` 조각 중복 처리
- `test_training_smoke.py` (`@pytest.mark.slow`): 로컬 e5-small로 전체/LoRA 각각 2 step 학습 → 저장 → `load_embedding_model`로 읽기 → 평가
- 마지막 검증: 실제 코퍼스 + 가짜 질문 약 200개로 split → train(짧게) → evaluate 전체 흐름

## 범위 밖
여러 GPU, 하이퍼파라미터 탐색, Hub 업로드, exp_003 연결, 질문 생성.
