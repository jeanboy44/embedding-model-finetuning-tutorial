# 실습 1 (5교시): 데이터 · 평가 · 기본 실험

장표: [5교시 데이터 · 평가 · 기본 실험](https://claude.ai/artifact/EzELtdX3t269cS4DHsmX3p). 0부 데이터 만들기(코퍼스 · 분할) → 1부 base 모델 평가(지표 → 학습 전 모델로 기준선) → 2부 1차 기준 모델 학습(기본 실험 001 한 바퀴) 순서로 본다. 교시가 끝나면 기준선(학습 전 e5, test R@5 0.545)과 기본 실험 001(0.744)이 한 표에 놓인다(실험 001, `experiments/001_baseline`).

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.
결과는 `experiments/results/lecture/05_data_eval/`에 쓰고, 받은 파일은 바꾸지 않는다. LLM 답변 단계는 이 강의에서 쓰지 않는다.

## 0. 준비 확인

3교시 `--fix`로 받은 데이터·모델·인덱스로 한다(2부에 쓰는 기본 실험 001 모델 포함). 5교시 줄이 준비됨이 아니면 `--fix`를 붙여 다시 돌린다. 조 단위 분할(`data/splits_article/`)은 받지 않고 질문에서 바로 만든다(몇 초, seed 42라 누가 만들어도 같다).

```shell
uv run python lecture/03_setup/01_doctor.py --quick
```

## 0부 · 데이터 만들기: 코퍼스와 분할

4교시에 질문을 만들어 봤다. 여기서는 그 질문이 찾아야 할 코퍼스와, 학습 · 평가를 나누는 분할을 본다.

### 1. 코퍼스: `01_corpus.py`

```shell
uv run python lecture/05_data_eval/01_corpus.py
```

받은 코퍼스 `data/processed/law_docs.json`을 `ragkit.data.load_corpus`로 읽고 네 단계로 본다(토큰 길이 재기 약 10초를 더해 20초 안쪽).

1. **규모.** 법령 54개(법률 53 · 시행령 53 · 시행규칙 34 · 대통령령 1 파일), 조문 11,723개, 검색 문서 25,967개를 센다. 테마별 법령 수와 검색 문서 수를 표로 보여 준다. tax가 10,239개로 가장 크고 electric이 1,141개로 가장 작다.
2. **검색 문서 한 건.** 장표 예시 `최저임금법_법률_제5조`의 필드(`id`, `title`, `category`, `theme`, `law_type`, `chapter`, `parent_id`, `paragraph`, `text`)를 그대로 찍는다. 모델에 넣는 글은 `ragkit.data.doc_text`가 만드는 "제목 + 줄바꿈 + 본문"이고, 앞의 `passage: ` 접두어는 `ragkit.embeddings.format_passages`가 코드에서 붙인다.
3. **항·호 분할.** `parent_id`가 같은 문서를 세어 나뉜 조문 2,596개가 조각 16,840개가 됐음을 본다. 한 조문이 가장 많이 나뉜 경우는 32조각이다. 예시 `전기사업법_법률_제2조`는 호를 묶어 5조각이 되고, 조각마다 같은 머리말을 앞에 붙여 문맥을 남긴다.
4. **길이 분포.** 글자 수와 e5 토크나이저(`ragkit.models.load_tokenizer`)로 잰 토큰 수의 중앙값 · 90% · 99% · 최대를 찍는다. 토큰 수 중앙값은 143, 99%는 509다. 512토큰 넘는 검색 문서가 247개(1.0%)이고, 나누기 전 조 단위 11,723개에서는 2,181개(18.6%)가 넘었다. 가장 긴 문서는 소득세법 시행령 제167조의3의 한 조각(3,651토큰)으로, 표처럼 더 못 나누는 문서가 남는다.

`--run`은 전기(electric) 테마만 `scripts/prepare_law_data.py`로 임시 폴더에 직접 만들어 받은 코퍼스와 비교한다(인터넷 필요, 수 초). legalize-kr 저장소를 임시 폴더에 받아 조문을 항·호로 나눈다. 받은 코퍼스의 electric 1,141개와 id 집합이 같고 본문까지 모두 같으면 재현된 것이다. 다르면 그 사이 법령이 개정된 것이다. 임시 폴더는 지우고 `data/processed/law_docs.json`은 바꾸지 않는다.

```shell
uv run python lecture/05_data_eval/01_corpus.py --run
```

코퍼스를 처음부터 만들려면 `uv run python scripts/prepare_law_data.py`(legalize-kr, 전체 테마 수 분). 받은 코퍼스를 덮어쓰므로 실습 중에는 하지 않는다.

### 2. 분할: `02_split.py`

```shell
uv run python lecture/05_data_eval/02_split.py
```

조 단위 분할 `data/splits_article/`을 `ragkit.training.load_splits`와 `split_meta.json`으로 읽고 세 단계로 본다(몇 초). 분할이 없으면 `ragkit split data/questions --unit article --test-ratio 0.15 --dev-ratio 0.1 --seed 42`를 먼저 돌려 만든다. 실험 001과 같은 분할이다.

1. **분할 표.** 질문 `data/questions`, 단위 article, seed 42, test 15% · dev 10% 목표를 적고 train 9,126 · dev 1,243 · test 1,863개(합계 12,232)를 표로 찍는다. 조는 4,389 · 610 · 942개, 법령은 세 분할 모두 54개다. 용도는 train 학습, dev epoch · 설정 고르기, test 최종 비교에 한 번만이다.
2. **조가 겹치지 않는가.** 정답 문서의 `ragkit.data.relevance_key`(조 id)로 train ∩ dev, train ∩ test, dev ∩ test가 모두 0개인지 확인한다. 같은 조의 항·호 조각은 제목이 같아 한쪽으로 묶는다. 이어서 train hard negative 16,432개 중 dev · test 정답 조의 문서 2,072개(12.6%)를 뺐고, 그래서 오답이 하나도 안 남은 train 질문 437개가 학습 예시에서 빠진다는 것을 본다.
3. **왜 조 단위인가.** test 질문 중 정답 조가 train에서도 정답으로 나온 비율을 잰다. 이 분할은 0.0%이고, 같은 질문을 질문 단위로 무작위로 섞어 같은 크기로 나누면 67.1%가 샌다. 법령 단위로 나누면 새지는 않지만 test 법령을 학습 때 한 조도 못 본다. 운영은 매뉴얼 전체로 학습하고 추가되면 다시 학습하므로, 조 단위가 현업 시나리오다.

`--run`은 같은 `ragkit split --unit article` 명령을 임시 폴더에 직접 돌려 지금 분할과 비교한다(몇 초). train · dev · test 질문 수와 조 수가 같고 세 파일이 바이트까지 같으면 "파일 같음 ✓"가 나온다. 같은 질문 · 같은 코퍼스 · 같은 seed면 누가 만들어도 같은 분할이 나온다. 임시 폴더는 지우고 `data/splits_article/`는 바꾸지 않는다.

```shell
uv run python lecture/05_data_eval/02_split.py --run
```

## 1부 · base 모델 평가: 지표와 학습 전 e5 = 기준선

### 3. 평가셋과 지표: `03_metrics.py`

```shell
uv run python lecture/05_data_eval/03_metrics.py
```

작은 예로 R@k · MRR · nDCG를 직접 계산해 보고, 학습 전 e5를 test 1,863개 전체로 잰다. 받은 인덱스 `data/processed/index/multilingual-e5-small.sqlite`를 다시 쓰므로 질문만 임베딩해 약 1분이다. 다섯 단계다.

1. **평가셋.** train · dev · test의 질문 · 조 · 법령 수를 다시 찍고, train과 test에 함께 든 조가 0개임을 확인한다. 검색 대상은 코퍼스 전체 25,967개 문서다. test 질문 유형은 keyword 399 · question 701 · situation 763이고, situation 질문 하나를 예로 보여 준다.
2. **지표를 손으로.** 질문 5개의 정답 순위 1 · 3 · 밖 · 7 · 12를 `ragkit.evaluation.question_metrics`에 넣어 R@1 · R@5 · R@10 · MRR@10 · nDCG@10을 질문마다 찍고 평균을 낸다. 평균은 R@5 0.40, MRR@10 0.295, nDCG@10 0.367이다. 주요 지표는 R@5다(화면 한 번에 보는 결과가 5개).
3. **판정 세 가지.** 항·호로 나뉜 조문 `전기사업법_법률_제2조`(5조각)를 골라 검색 결과 순서를 가정하고 `ragkit.evaluation.first_rank`로 doc 판정(정답 문서와 정확히 같아야 함) 3위, article 판정(같은 조의 다른 조각도 정답) 1위를 비교한다. 복수 정답 판정 파일 `data/labels/dev.jsonl`은 질문 641개를 판정해 다른 정답을 인정한 질문이 44개(6.9%)다. multi 판정은 정답 또는 인정한 다른 정답이 나오면 정답으로 친다. test에는 판정 파일이 없어 multi = doc이다.
4. **학습 전 e5 평가.** `ragkit evaluate intfloat/multilingual-e5-small --split test --splits data/splits_article`을 부른다. 결과는 `experiments/results/lecture/05_data_eval/multilingual-e5-small_test.json`에 쓴다.
5. **결과 읽기.** doc 판정 R@1 0.284 · R@5 0.545 · R@10 0.630 · MRR@10 0.392 · nDCG@10 0.449, article R@5 0.627이 나온다(실험 001의 "000 학습 전 e5"와 같다). 질문 유형별 R@5는 keyword 0.739 / question 0.573 / situation 0.417로, 상황 설명형에서 작은 모델이 무너진다. 테마별로는 consumer 0.435가 가장 낮고 electric 0.753이 가장 높다. 마지막에 5위 안에 못 든 situation 질문 445개 중 3개를 뽑아 1위로 나온 문서와 정답 순위를 보여 준다.

`--run`은 test에서 300개를 seed 42로 뽑아 `ragkit evaluate`를 임시 폴더에 직접 돌린다(약 30초). 이어서 dev는 복수 정답 판정이 있는 81개만 골라 `--labels data/labels/dev.jsonl`을 붙여 다시 평가하고, doc R@5가 multi 판정에서 오르는 것을 본다(판정 파일은 옛 dev로 만들어 지금 dev와 일부만 겹친다). 표본 300개의 R@5는 전체 test 값과 조금 다르다.

```shell
uv run python lecture/05_data_eval/03_metrics.py --run
```

## 2부 · 1차 기준 모델 학습: 기본 실험 001 한 바퀴

### 4. 학습 원리와 파일럿: `04_baseline_train.py`

```shell
uv run python lecture/05_data_eval/04_baseline_train.py
```

가설 → 설정 → 파일럿 → 학습 → dev로 epoch 선택 → test 1회. 학습 원리(MNRL, NO_DUPLICATES)와 실험 001의 설정 파일, 받은 모델 `models/finetuned/001_baseline`의 학습 기록을 본다(1분 안쪽). 기본 모드는 네 단계다.

1. **학습 데이터 한 줄.** train 질문을 `ragkit.data.filter_questions`로 거른 뒤 situation 질문 하나의 질문 · 정답 조문 · hard negative 제목을 찍는다. hard negative는 질문을 만들 때 같은 법에서 고른 비슷한 조문이다. train 9,126개 중 hard negative가 남은 8,689개가 학습에 들어간다.
2. **MNRL과 in-batch negative.** 법령이 서로 다른 질문 4개를 골라 배치를 만들고, 학습 전 e5로 `ragkit.training.in_batch_scores`를 불러 4×8 코사인 점수표(정답 4 + hard negative 4)와 MNRL 손실 1.514를 찍는다. 점수가 0.78~0.88에 몰려 있고 대각선(자기 정답)이 뚜렷이 크지 않다. 실제 학습은 배치 32라 질문 하나에 in-batch 오답 31개 + hard negative다.
3. **NO_DUPLICATES.** train 질문 9,126개 중 정답 조문을 다른 질문과 같이 쓰는 질문이 3,466개(38%)다. 가장 많은 예는 교통사고처리 특례법 제3조 제2항 조각으로 질문 5개가 같은 정답이다. 한 배치에 들어가면 서로의 정답을 오답으로 배우므로 `BatchSamplers.NO_DUPLICATES`로 막는다.
4. **설정 파일과 학습 기록.** `experiments/001_baseline/config.yaml`의 training 항목(배치 32, cached_mnrl, mini 16, 3 epoch, lr 2e-5, warmup 0.1, 길이 512, negative 1, seed 42)을 찍고, 받은 모델의 `train_meta.json`에서 학습 39분, 파라미터 117,653,760개 전부 학습, best epoch 3, dev R@5 학습 전 0.540 → 0.697 / 0.710 / 0.717을 읽는다. epoch는 dev로 고르고 test는 보지 않는다.

`--run`은 파일럿이다. 실험 001 설정 그대로 `ragkit train`을 `--output-dir` 임시 폴더, `--limit 960`(배치 32 × 30 step), `--max-steps 30`, `--no-dev-eval`로 돌린다(`uv sync --extra train` 필요, 모델 로드 포함 2~3분). step당 시간이 찍히고 같은 속도로 3 epoch면 몇 분일지 어림한다. 이어서 2단계의 같은 배치 4개를 파일럿 모델로 다시 재서 점수표와 손실, 질문별 여유(정답 점수 − 가장 높은 오답 점수)를 학습 전과 비교한다. 30 step만으로도 0.78~0.88에 몰려 있던 점수가 넓게 퍼진다. 확인할 것은 손실이 줄었나, 메모리가 버텼나, step당 시간으로 전체 시간을 어림했나 세 가지다. 결과 모델은 임시 폴더에 두었다가 지운다(약 470MB).

```shell
uv run python lecture/05_data_eval/04_baseline_train.py --run
```

전체 학습(M4 Pro 39분)은 강사가 미리 한 모델(`models/finetuned/001_baseline`)을 쓴다. test 결과(R@5 0.545 → 0.744, 실험 001)는 장표로 보고, 6교시에 판정한다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_corpus.py` | 코퍼스 규모, 문서 한 건의 필드, 항·호 분할, 글자·토큰 길이 분포 |
| `02_split.py` | 조 단위 분할 표(없으면 만든다), 조 교집합, 오답에서 뺀 문서, 질문 단위로 나눌 때 새는 비율 |
| `03_metrics.py` | 평가셋(조 단위 test), 지표 직접 계산, doc · article · multi 판정, 학습 전 e5 test 평가, 유형·테마별, 틀린 질문 예 |
| `04_baseline_train.py` | 학습 데이터 한 줄, in-batch 점수표와 MNRL 손실, NO_DUPLICATES 근거, 실험 001 설정 파일, 받은 모델의 학습 기록, `--run` 파일럿 |

- 직접 전체 학습을 하려면 `uv run ragkit train experiments/001_baseline/config.yaml --splits data/splits_article --output-dir models/finetuned/my_001`처럼 출력 폴더를 바꿔서 한다(약 40분, 받은 모델을 덮어쓰지 않게).
