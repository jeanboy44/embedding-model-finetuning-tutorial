# 012 remined_negatives — 2. 실험 계획 (v2, 검토 반영)

앞 단계 문서: `00_error_analysis.md`, `01_research.md`. 이 계획서의 dev 숫자는 모두 두 문서와 `runs/001_baseline_seeds/`의 집계 파일에서 옮겼다. dev 질문별 결과 파일(`eval_dev*.json`, `topk_dev_a.jsonl`)은 열지 않았다.
채굴 숫자(풀 크기, negative 0개 수, 관계 분포, 버린 후보)는 이 계획을 쓰면서 **train 분할로만** 시험 채굴해 얻었다. 출처는 `runs/012_remined_negatives/work/plan/full_test/{mine_stats.json, check.out}`이다. 시험 채굴은 train 질문만 임베딩했고 학습 결과가 아니다. 검토 두 개(`reviews/plan_ir_expert.md`, `reviews/plan_methodology_expert.md`)가 이 숫자를 다시 세어 확인했다.

v1 → v2에서 바뀐 것:
- 가설 문구를 실제 선택 규칙(정답보다 아래인 문서만)에 맞췄다.
- A3을 이번 바퀴 **탐색 전용**으로 내렸다. A3_s43 · C_s42는 없앴다.
- 메커니즘 해석 기준, 보조 지표, 학습 뒤 digest 기대값을 미리 적었다.
- "실패"의 정의를 적었다.
- `run_queue.py`의 실패 동작 서술을 고쳤다.
- 판정 표본을 200쌍으로 늘렸다.

자세한 것은 맨 끝 "검토 반영" 표에 있다.

## 실험 이름

**`remined_negatives`**. 폴더는 `experiments/012_remined_negatives/`, `runs/012_remined_negatives/`로 바뀐다. 이 문서의 경로와 설정 파일은 바뀐 이름 기준으로 적었다.

## 가설

**001의 학습 설정과 학습 질문 집합(8,689행)은 그대로 두고, 질문마다 붙은 hard negative만 바꾼다.**
- 지금 negative: LLM이 같은 파트에서 고른 문서(같은 조 조각 50%, 다른 법 0%).
- 바꿀 negative: 001이 직접 캔 문서 1개. 조건은 평가 쪽 조가 아니고, **정답보다 점수가 낮으면서 정답 점수의 95% 미만**이고, 풀 안 1~10위 중 가장 높은 문서다(없으면 11~30위).
- 예상: 정답 바로 아래에 붙어 있는 다른 법 · 같은 법 다른 조 문서와 정답 사이의 간격을 벌리게 되어, dev R@5(doc 판정)가 공용 대조군 seed 3개 평균보다 **0.02 이상** 오른다.

틀릴 수 있는 지점:
- (a) Δ < 0.02이거나 seed 범위 안이면 기각한다.
- (b) dev-b에서 방향이 반대면 기각한다.
- (c) 메커니즘("다른 법 · 법률↔시행령 혼동이 줄어서 올랐다")은 아래 "해석 규칙"의 수치 기준을 넘을 때만 쓴다.

근거:
- 실패 1위 오답의 37%가 다른 법이고 32%가 법률↔시행령인데, 기존 HN은 다른 법 0%, 법률↔시행령 3%다. 실패 1위가 그 질문의 HN인 경우도 3.3%뿐이다(`00` "학습 데이터 쪽").
- 001로 캔 상위 오답의 관계 분포는 dev 실패 분포와 거의 같다(`00` [13]).
- 옛 분할에서 같은 방법이 +0.041(1개 변형 +0.031)이었다(`01` 근거 표, 002 A · C).
- 리서치는 기대 크기를 +0.02~0.04로 낮춰 잡았다(`01`).

**가설이 닿지 못하는 범위 (검토 반영)**
1. **95% 컷은 정답보다 위에 있는 후보를 모두 버린다.** 그래서 이 데이터가 주는 신호는 주로 "이미 정답이 이기는 질문에서 2등 근처 문서와 간격 벌리기"다.
   - 학습 대상 8,689문항 중 45.7%(3,967)는 풀 상위 10 안에 정답보다 높은 후보가 있었고, 모두 버렸다. 버린 후보 17,261개의 관계는 다른 법 44%, 법률↔시행령 27%다.
   - 001이 train에서 top-5 밖인 1,811문항이 받는 negative의 사유는 top 392 · fill 805 · llm_fallback 498 · excluded 116이다. 즉 틀리는 질문의 대부분은 상위 10위의 혼동 문서를 negative로 받지 못한다(`full_test/mine_stats.json`의 `dropped_by_margin_top10`, `train_failures_reason`).
2. dev 실패 1위 문서의 22%는 평가 쪽 조라 negative로 줄 수 없다(`00`, `01` 함정 표). 이 혼동은 train 쪽 비슷한 혼동에서 일반화되기만 기대할 수 있다.

**부가 질문 (탐색 전용, 채택 대상 아님)**: negative를 3개로 늘리면(A3) 1개(A1)보다 더 오르는가? A3은 seed 1개만 돌려 보고만 한다. 채택하지 않고, 결과는 다음 바퀴 후보의 근거로만 쓴다(검토 M3).

## 제약 확인

| constraints.md 항목 | 이 계획에서 지키는 방법 |
|---|---|
| 1 · 모델은 e5-small 하나, 다른 임베딩 모델 금지 | 학습 시작 모델은 `intfloat/multilingual-e5-small`이다. 채굴 모델은 001(`models/finetuned/001_baseline`)로, 같은 e5-small을 학습한 것이다 |
| 1 · reranker · BM25 · LLM 확장 · 검색 시 LLM 금지 | 쓰지 않는다. 가짜 negative는 정답 점수비(95%)로만 거른다 |
| 1 · 크기 · 차원(118M, 384) 그대로 | 모델 · 손실 · 길이를 바꾸지 않는다. 바꾸는 것은 학습 데이터의 negative뿐이다 |
| 1 · 판정은 Claude 서브에이전트, 외부 API 금지 | 데이터를 바꾸는 판정은 없다. 기록용 표본 200쌍(L9)만 Claude 서브에이전트가 판정한다 |
| 2 · 분할은 `data/splits_article` 고정 | 학습 명령은 `--splits data/splits_article` 그대로다(`run_queue.py`가 고정한다). 새 negative는 설정의 `negatives_file`로 넣는다 |
| 2 · **새 학습 데이터 재료는 train 조에서만** | 바뀌는 재료는 negative 문서뿐이다. 후보 풀에서 **dev · test 정답 조 1,552개에 속한 문서 4,388개를 모두 뺀다**(풀 21,579 / 코퍼스 25,967). 풀 제한 전에는 상위 30의 19.6%가 평가 쪽이었다. 질문 · 정답은 train 그대로다. 질문 없는 조의 문서는 풀에 남긴다(고정한 것 표 참고) |
| 2 · 새 학습 질문의 근사 중복 검사 | 새 질문이 없다. 학습 질문은 001과 같은 8,689개이고 L2 · L10으로 확인한다 |
| 2 · 오답 분석은 dev-a에서만, dev-b는 집계만 | 계획 에이전트는 dev 질문별 파일을 열지 않았다. 결과 단계의 질문 원문 확인은 dev-a에서만 한다. dev-b는 스크립트로 집계한 숫자만 본다 |
| 2 · run · epoch 선택은 dev로만 | epoch은 학습 중 dev R@5로 고르고(001과 같음), 판정은 공식 dev 평가와 `decide.py`로 한다 |
| 2 · test는 바퀴 안에서 쓰지 않음 | 어떤 명령도 `--split test`를 쓰지 않는다. `mine.py`는 풀에서 뺄 정답 조 목록을 만들 때만 `test.jsonl`을 읽는다. L7로 점검한다. `01_research.md`의 005 · 004 test 숫자는 옛 분할 값이라 이 계획의 근거로 쓰지 않는다(검토 m9) |
| 2 · 분할이 바뀐 뒤 첫 바퀴는 기준 모델 재측정 | 재현 점검은 이미 통과했다(학습 전 0.5398, 001 0.7168, `runs/012_remined_negatives/timeline.md`) |
| 2 · 판정 파일은 현재 분할용만 | `--labels` 없이 doc 판정만 한다 |
| 3 · 대조군 seed 3, 같은 코드 · 설정 · 스케줄 | 공용 대조군 `runs/001_baseline_seeds/{s42,s43,s44}`를 다시 쓴다(아래 재사용 확인) |
| 3 · 채택 후보 seed 2 이상, seed 평균끼리 비교 | 채택 후보는 A1 하나다. 처음부터 s42 · s43 두 개를 학습한다 |
| 3 · Δ ≥ 0.02 · Δ > seed 범위 · CI 하한 > 0 · Holm | `decide.py` 기본값에 조건 하나를 더해 더 엄격하게 한다(합격선) |
| 3 · 표준오차 √(p(1−p)/n) | dev 1,243문항, p=0.717이면 ±0.0128이다. dev-b 621문항, p=0.725이면 ±0.0179다 |
| 3 · 법령 평균 · 나빠진 법령 0 | `decide.py` 조건 4 |
| 3 · dev-b Δ 따로, 반대면 개선 아님 | `decide.py` 조건 5. dev-b 대응 검정 CI도 판정표에 같이 적는다 |
| 3 · 두 가지가 같이 바뀌면 하나씩만 다른 run | A1은 출처 하나만 바꾼다(개수 1, 질문 집합 그대로). A3은 A1에서 개수만 바꾸며 탐색 전용이다. 437문항 되살리기는 하지 않는다 |
| 3 · 계산량이 다르면 기록 · 맞춘 대조 | A1은 step 수가 대조군과 같다(816 step, 검토 m10 표에서 확인). A3은 시간이 약 1.7배라 채택 대상에서 뺐다. 그래서 계산량 맞춤 대조가 필요 없다 |
| 4 · 학습 run 8개 이하 | 파일럿 1 + 본 3 = **4 run**이다. 대조군은 재사용하므로 0이다. 다시 돌리기(아래 "실패" 정의)는 run당 1회, 최대 2 |
| 4 · 판정은 300쌍 시범부터 | 데이터를 바꾸는 판정은 없다. 기록용 200쌍은 시범 규모 안이다 |
| 4 · 본 학습 전 파일럿 | `pilot_A3`(메모리가 가장 큰 설정)를 먼저 단독으로 돌린다 |
| 5 · 재현 점검 | 통과했다 |
| 5 · 채택 모델은 배포 형태로 dev 한 번 | A1을 채택하면 가지치기 + INT8로 dev를 잰다(오케스트레이터) |
| 5 · 에이전트별 전용 폴더 | 계획 에이전트 파일은 `runs/012_remined_negatives/work/plan/`에만 있다 |
| 5 · runs · timeline · MLflow 기록 | 아래 로깅 |
| 5 · 커밋 금지, 덮어쓰기 금지 | 새 데이터는 `runs/012_remined_negatives/data/`, 모델은 `models/finetuned/012_remined_negatives_<run>`에 둔다 |

### 공용 대조군 재사용 확인

| 조건 | 확인 | 결과 |
|---|---|---|
| 학습 코드 | `git log --oneline -- src/ragkit/training/`의 마지막 커밋은 `5ae5c90`(001을 조 단위로 다시 학습)이다. `git diff 5ae5c90 -- src/ragkit/training/`은 비어 있다. `git diff 5ae5c90 --stat -- src/ pyproject.toml uv.lock`도 비어 있다(2026-10-11 03:05 확인, 방법론 검토도 다시 확인) | 같음 |
| 분할 | 세 seed 모두 `--splits data/splits_article`이다. s42 `train_meta.json`의 digest는 train d9fec2ea3431(n 9,126), dev 781b8ee713a4(n 1,243)이고 현재 분할과 같다. s42 가중치 시각(02:35)은 5ae5c90 커밋(01:54) 뒤다 | 같음 |
| 설정 | s43 · s44 `config.yaml`은 001에서 `output_dir` · `seed`만 다르다 | 같음 |
| 바꾸는 것 | 후보 run은 `negatives_file`(A1 · A3), `num_negatives: 3`(A3), `seed: 43`(A1_s43)만 더한다 | 의도한 차이뿐 |

결론: **재사용한다.** 단서가 세 개 있다.
1. s42 학습 당시 커밋되지 않은 변경이 없었는지는 확인할 수 없다. src가 같아 같은 분포로 본다.
2. s43 · s44의 `eval_dev.json` · `eval_dev_b.json`이 생겨야 `decide.py`를 돌릴 수 있다. 실패하면 공용 대조군 몫으로 다시 학습하고, 012 예산에는 넣지 않는다.
3. s43 · s44를 돌리는 프로세스는 `run_queue.py`가 바뀌기 전 코드(실패해도 다음으로 넘어가던 버전)로 떠 있을 수 있다. 이 사실을 `timeline.md`에 적는다(검토 m8).

## 고정한 것 (사전 등록)

| 항목 | 값 | 근거 |
|---|---|---|
| 학습 설정 | `experiments/001_baseline/config.yaml` 그대로: CachedMNRL mini 16, 배치 32, lr 2e-5, warmup 0.1, epoch 3(dev R@5로 선택), 길이 512, attention dropout 0, NO_DUPLICATES | 대조군과 같은 스케줄이어야 Δ가 negative 효과다 |
| 학습 질문 집합 | 001과 같은 8,689개. 001에서 negative 0개로 빠졌던 437문항은 이번에도 뺀다(`negative_ids: []`) | 질문 집합이 바뀌면 출처 효과와 섞인다 |
| 채굴 모델 | 001 = `models/finetuned/001_baseline`(대조군 s42와 같은 가중치). 채굴에만 쓰고, 모든 run은 학습 전 e5에서 시작한다 | 00 "001로 캔 후보 분포 ≈ dev 실패 분포" |
| 채굴 입력 | 001로 train 9,126문항 전체를 임베딩한다. 코퍼스는 001 평가 인덱스(`data/processed/index/001_baseline-20261010173529.sqlite`)를 쓴다 | 시험에서 train R@5 0.802 |
| **후보 풀** | 코퍼스 25,967개에서 dev · test 정답 조(relevance_key 1,552개)의 문서를 뺀 21,579개. 정답과 `related_ids`도 뺀다. 질문 없는 조의 문서는 남긴다 | 평가 정답 조가 학습 오답으로 나오면 평가가 움직인다(lessons). 질문 없는 조는 어느 평가에도 정답이 아니다. 001 데이터(`drop_heldout_negatives`)와 같은 정의다 |
| 순위 범위 | 풀 안 1~10위에서 점수 순으로 고르고, 모자라면 11~30위(`fill`), A3에서 그래도 모자라면 반복(`repeat`) | NV-Retriever top-10, 002 |
| 가짜 negative 컷 | (후보 점수 / 정답 점수) ≥ 0.95면 버린다. 정답보다 높은 후보도 여기서 모두 버려진다. 판정은 없다 | NV-Retriever TopK-PercPos 95%, 002 A. 버리는 쪽의 비용은 위 "닿지 못하는 범위"와 L9로 기록한다 |
| negative 수 | A1 = 1개(대조군과 같음), A3 = 3개(탐색) | 01 우선순위 |
| **negative 0개 규칙** | 001에서 이미 0개였던 437문항은 `[]`(`excluded_as_001`). 그 밖에 30위까지 전부 95% 컷에 걸린 질문은 **그 질문의 기존 LLM negative로 대체**한다(`llm_fallback`, train 쪽만 남은 것, A3이면 반복해 3개). 이 규칙으로 빠지는 질문은 0개다 | lessons. 시험에서 대체는 499문항(학습 행의 5.7%)이었고, 그중 대부분(75%)은 정답이 30위 밖이다(검토 IR 사소 4) |
| 위임 연결 짝 | 빼지 않는다. 수를 기록하고 L9로 가짜 비율을 잰다 | 00 라벨 표본 partial 2/40 |
| 판정 | doc 기준, `--labels` 없음 | 제약 2절 |
| 대응 검정 기준 | 대기열 control 열 = `../001_baseline_seeds/s42` | run_queue 형식. s43 · s44 대비 검정도 보고한다(검토 m1) |
| **"실패"의 정의** | 다시 돌릴 수 있는 실패는 다음 넷뿐이다: 프로세스 비정상 종료(예외), 메모리 부족(OOM), 학습 손실 NaN, `eval_dev.json` 미생성. **점수가 낮거나 이상해 보이는 것은 실패가 아니며 다시 돌리지 않는다.** 다시 돌린 run은 같은 seed · 같은 설정으로 처음 run을 대체하고, run당 1회까지다. 사실과 원인은 `timeline.md`에 남긴다 | 검토 M2. MPS는 비결정적이라 다시 돌리기가 사실상 seed를 하나 더 뽑는 일이 된다 |

### 새 학습 데이터 만드는 법

스크립트는 계획 단계에서 시험을 마쳤다(`runs/012_remined_negatives/work/plan/mine.py`, `check_data.py`). 폴더 이름을 바꾼 뒤 저장소 루트에서 실행하고, 실제로 친 명령을 `runs/$E/data/mine.command.sh`에 남긴다.

```bash
E=012_remined_negatives
# 1) 001로 train 전체 임베딩 · 상위 100 (CPU 약 40초)
uv run python runs/$E/work/error_analysis/dump_topk.py models/finetuned/001_baseline train runs/$E/data/mine_input
# 2) 채굴 → runs/$E/data/{mined_candidates.jsonl, mine_stats.json, judge/, A1/, A3/} (약 1분)
uv run python runs/$E/work/plan/mine.py runs/$E/data/mine_input runs/$E/data
# 3) 누수 · 일치 점검 → runs/$E/data/{A1,A3}/check.json, 하나라도 실패하면 종료 코드 1 → 학습을 띄우지 않는다
uv run python runs/$E/work/plan/check_data.py runs/$E/data/A1 runs/$E/data/A3
# 4) run 설정 복사
for r in pilot_A3 A1_s42 A1_s43 A3_s42; do mkdir -p runs/$E/$r; cp runs/$E/work/plan/configs/$r/config.yaml runs/$E/$r/; done
```

- **학습에는 실험 폴더에서 다시 만든 파일을 쓴다.** 그 sha256과 학습 뒤 digest 기대값(L10)을 학습 전에 `data_digest.json`에 고정한다.
- 다시 만든 sha가 아래 시험값과 다르면 차이 나는 행 수를 기록한다. 그리고 digest 기대값을 다시 계산해 학습 전에 고친다(검토 m7). 계산은 `ragkit.data.questions_digest`를 `_replace_negatives` 적용 뒤의 train 질문에 돌린다.

| 경로 | 내용 |
|---|---|
| `runs/012_remined_negatives/data/<run>/negatives.jsonl` (A1 · A3) | qid · negative_ids · reasons(top / fill / repeat / llm_fallback / excluded_as_001). **학습이 실제로 읽는 파일**이다(`negatives_file`) |
| `runs/012_remined_negatives/data/<run>/train.jsonl` | `data/splits_article/train.jsonl`과 같은 9,126행 · 같은 순서이고 `hard_negative_ids`만 바뀐다. 기록 · 점검용이다(L3로 학습 입력과 같음을 확인) |
| `runs/012_remined_negatives/data/<run>/dev.jsonl` | `data/splits_article/dev.jsonl` 그대로 복사한 것(sha256 일치). test는 복사하지 않는다 |
| `runs/012_remined_negatives/data/mined_candidates.jsonl` | 질문마다 풀 안 상위 30: id · 순위 · 점수 · 정답 대비 점수비 · 관계 6유형 |
| `runs/012_remined_negatives/data/mine_stats.json` | 풀 크기, 풀 제한 전 평가 쪽 비율, 선택 사유, negative 0개 · 대체 수, 관계 분포, 위임 연결 짝 수, **정답보다 높아 버린 후보 · 그런 질문 비율 · 관계**, **train 실패 질문의 사유 분포**, **질문당 서로 다른 negative 수 · 반복 행 수**, **평가 쪽 정답 문서와 코사인 ≥ 0.98인 negative 수**, sha256 |
| `runs/012_remined_negatives/data/judge/{input,key}.jsonl` | L9 판정 200쌍. `input`에는 점수 · 순위 · 종류를 넣지 않고 순서를 섞는다. `key`는 집계용이다 |

`run_queue.py`는 `--splits data/splits_article`을 고정하고, `ragkit train`은 `--splits`를 두 번 주면 거부한다(직접 확인: "Parameter --splits specified multiple times"). 그래서 새 데이터 폴더를 `--splits`로 넘기지 않고 학습 코드가 이미 지원하는 `negatives_file`로 negative만 바꾼다.

시험 채굴 결과 (`runs/012_remined_negatives/work/plan/full_test/`):

| 항목 | A1 | A3 (탐색) |
|---|---|---|
| 학습 예시(행) | 8,689 | 8,689 |
| 선택 사유 | top 7,341 · fill 849 · llm_fallback 499 · excluded_as_001 437 | top 21,655 · fill 2,849 · llm_fallback 1,497(499문항) · repeat 66 · excluded 437 |
| 관계(질문별 고른 문서) | 다른 법 같은 테마 22.3% · 같은 법 같은 종류 21.8% · 법률↔시행령 연결 없음 16.7% · 같은 조 조각 13.9% · 다른 법 다른 테마 12.0% · 위임 연결 8.8% · 기존 LLM 4.6% | 다른 법 같은 테마 24.0% · 같은 법 같은 종류 22.3% · 연결 없음 17.7% · 다른 법 다른 테마 14.0% · 같은 조 조각 12.2% · 위임 연결 7.4% · 기존 LLM 2.6% |
| 다른 법 비율 (기존 HN 0%) | 34.2% | 37.9% |
| 같은 조 조각 비율 (기존 HN 50%) | 13.9% | 12.2% |
| 위임 연결 짝 | 760 | 1,864 |
| 서로 다른 negative 문서 | 6,121 (최다 10회, 쏠림 없음, IR 검토) | 11,470 |
| 질문당 서로 다른 negative 수 | 1개 8,689 | 3개 8,146 · 2개 380 · 1개 163 (반복이 들어간 행 543, 6.2%) |
| 평가 쪽 정답 문서와 코사인 ≥ 0.98 | 3 | 4 |
| 정답보다 높아 버린 후보 (풀 상위 10) | 17,261개, 3,967문항(45.7%) | 같음 |
| train 실패 1,811문항의 사유 | top 392 · fill 805 · llm_fallback 498 · excluded 116 | 같음 |
| negatives sha256 | 58ce0cb2…cf6de | d4c897b9…3c856 |
| 학습 뒤 `data.train.digest` 기대값 | **2c3d054b41a1** | **6ce401a22a7a** |
| check_data | 8개 항목 모두 통과 | 8개 항목 모두 통과 |

## run 구성

| run | 바꾸는 것 (001 대비) | seed | 데이터 | 예상 시간 | 역할 |
|---|---|---|---|---|---|
| 대조군 s42 · s43 · s44 | 없음 (공용 `runs/001_baseline_seeds/`) | 42 · 43 · 44 | 기존 LLM negative 1개, 8,689행 | 0 (재사용) | 대조군. s43 · s44는 학습 중 |
| pilot_A3 | A3과 같음, 30 step · 300문항 · dev 평가 없음 | 42 | `data/A3/negatives.jsonl` | 약 3~5분 | 파일럿 (메모리 최대 설정) |
| **A1_s42** | negative 출처: LLM → 001 채굴 1개 | 42 | `data/A1/negatives.jsonl`, 8,689행 | 약 40분 + 평가 약 5분 | 채택 후보 |
| **A1_s43** | A1, 두 번째 seed | 43 | 같음 | 약 40분 + 평가 | 채택 후보 (처음부터 두 seed) |
| A3_s42 | A1에서 negative 개수만 1 → 3 | 42 | `data/A3/negatives.jsonl`, 8,689행 | 약 65분 + 평가 (1.7배, 002 비율) | 탐색 전용, 채택 안 함 |

예산: 4 run(제약 8 이하). 다시 돌리기는 위 "실패" 정의에 걸릴 때만, run당 1회, 최대 2다. 세 번째 seed나 조건부 run은 없다.

### queue.tsv

최종 대기열의 사본은 `runs/012_remined_negatives/work/plan/queue.tsv`에 있다. `runs/012_remined_negatives/queue.tsv`에 탭 구분으로 넣는다.
- **1단계**: 머리글과 파일럿 행만 올리고 돌린다.
- **2단계**: 파일럿 통과를 확인한 뒤 본 run 3행을 덧붙여 다시 띄운다.
- `run_queue.py`는 run 하나가 실패하면 **대기열 전체를 멈춘다**. 남은 run은 "대기" 그대로 남는다. 그러니 한 번에 다 올려도 파일럿이 실패하면 본 학습은 돌지 않는다. 그래도 파일럿 결과(step당 시간 · 메모리)를 보고 본 run 시간을 다시 추정하려고 두 단계로 나눈다.
- 공용 대조군 s43 · s44가 끝난 뒤 띄운다(GPU 하나).

```
run	status	start	end	control	model_dir	train_args
pilot_A3	대기				models/finetuned/012_remined_negatives_pilot_A3	--max-steps 30 --limit 300 --no-dev-eval
A1_s42	대기			../001_baseline_seeds/s42	models/finetuned/012_remined_negatives_A1_s42	
A1_s43	대기			../001_baseline_seeds/s42	models/finetuned/012_remined_negatives_A1_s43	
A3_s42	대기			../001_baseline_seeds/s42	models/finetuned/012_remined_negatives_A3_s42	
```

순서는 채택 후보 A1을 먼저, 탐색 A3을 마지막에 둔다. 예산이 모자라면 A3_s42를 뺀다. run 폴더의 `config.yaml`은 `runs/012_remined_negatives/work/plan/configs/<run>/config.yaml`을 복사한다. 001과 다른 줄에는 `# 바꿈` 주석이 있고, 모두 `load_train_config`로 읽힌다.

**파일럿 통과 기준**
- 손실이 NaN이 아니고, 30 step 동안 줄어드는 방향이다.
- 메모리 부족이 없다.
- "채굴 결과에 없는 train 질문" 오류가 없다.
- step당 시간으로 A1 · A3 본 학습 시간을 다시 추정해 `timeline.md`에 적는다.

## 누수 점검

| # | 점검 | 수행 방법 | 기록 위치 |
|---|---|---|---|
| L1 | negative에 평가 쪽 조 문서 · 정답 · related 문서가 없음 | `mine.py`가 후보 생성 직후와 선택 직후 assert하고(llm_fallback 포함), `check_data.py`가 다시 센다(0이어야 함). 풀 제한 전 평가 쪽 비율(시험 19.6%)을 남긴다. 기록용으로 평가 쪽 정답 문서와 001 코사인 ≥ 0.98인 negative 수도 남긴다(시험 A1 3 · A3 4, 검토 m11). 이 수는 판정 조건이 아니다 | `data/mine_stats.json`, `data/<run>/check.json` |
| L2 | 학습 질문은 그대로, negative만 바뀜 | `check_data.py`가 원 `train.jsonl`과 행 수 · 순서 · (query, positive_id) · 다른 필드 전부를 비교한다 | `check.json` |
| L3 | 학습이 실제로 읽는 데이터 = 기록한 데이터 | `check_data.py`가 학습 코드의 `filter_questions` → `_replace_negatives` → `to_examples`를 그대로 불러 `train.jsonl` 사본과 비교한다 | `check.json` |
| L4 | dev 사본이 원본과 같음 | sha256 비교 | `check.json`, `data_digest.json` |
| L5 | 채굴에 dev · test 질문을 쓰지 않음 | 채굴 명령은 `dump_topk.py ... train`뿐이다. `mine_input/`에 `query_emb_train.npy`만 있는지 `ls`로 확인한다 | `data/mine.command.sh`, `timeline.md` |
| L6 | 채굴 모델에 담긴 평가 정보 | 001은 train으로만 학습했지만 epoch 선택에 dev를 썼다. 대조군도 같은 영향을 갖는다. 채굴 모델 = 대조군 s42라서 s42 대비 대응 검정이 약간 유리할 수 있다. 그래서 s43 · s44 대비 검정도 계산한다(합격선 보고 항목) | `04_results.md` 한계 |
| L7 | 바퀴 안에서 test를 쓰지 않음 | 바퀴 끝에 `grep -rn -- "--split test\|test.jsonl" runs/012_remined_negatives/*/command.sh runs/012_remined_negatives/data/mine.command.sh`의 결과가 비어야 한다 | `timeline.md`에 grep 결과 한 줄 |
| L8 | 역할 분리 | 계획 에이전트는 dev 질문별 파일을 열지 않았다. 결과 단계에서 질문 원문은 dev-a만 열고, dev-b는 스크립트 집계 숫자만 본다 | `04_results.md` |
| L9 | 가짜 negative 비율 (기록용, 데이터 안 바꿈) | `mine.py`가 만든 200쌍을 Claude 서브에이전트 하나가 full / partial / no로 판정한다(00 라벨 표본과 같은 기준). 구성은 A1 선택(top · fill) 100, A1 위임 연결 50, **버린 "정답보다 위" 후보 50**이다(검토 IR-1). 프롬프트에는 질문 · 모범 답 · 정답 조문 · 후보 조문만 넣는다. 집계는 종류별, 그리고 점수비 구간(≥ 0.92 / < 0.92)별로 낸다. A1 선택에서 full이 10% 이상이면 결과 해석의 한계로 적는다. "정답보다 위" 후보의 no 비율은 다음 바퀴(컷 바꾸기)의 근거로 남긴다 | `runs/<실험>/work/judge/{input,output}.jsonl`, `judge.out` |
| L10 | 질문 집합 · 행 수가 001과 같음 | 학습 전에는 `check_data.py`(학습 예시 8,689, 질문 집합 = 001)로 본다. **학습 뒤**에는 `train_meta.json`에서 `train_examples == 8689`, `data.train.n == 9126`, `data.train.digest`가 A1 run이면 **2c3d054b41a1**, A3 run이면 **6ce401a22a7a**인지 본다. digest는 negative를 바꾼 뒤의 질문으로 계산하므로 001의 d9fec2ea3431과 같을 수 없다(검토 M1). 다시 채굴해 sha가 바뀌었으면 학습 전에 고친 기대값을 쓴다 | `check.json`, `summary.tsv`, `data_digest.json` |

`check_data.py`가 하나라도 실패하거나 L10 학습 뒤 값이 다르면, 다음 run을 띄우지 않고 원인을 `timeline.md`에 적는다.

## 개선 판정 기준 (합격선)

결과를 보기 전에 정한다. 계산은 `decide.py`로 한다.

```bash
uv run python .claude/skills/experiment-loop/scripts/decide.py 012_remined_negatives \
  --control ../001_baseline_seeds/s42 ../001_baseline_seeds/s43 ../001_baseline_seeds/s44 \
  --candidate A1=A1_s42,A1_s43 A3=A3_s42
```

- **채택 후보는 A1 하나다.** A3은 탐색으로 계산만 하고, 조건을 다 넘어도 채택하지 않는다. `decide.py`는 seed 2개 미만이면 통과를 주지 않는다.
- Holm 보정의 family는 {A1, A3} 두 개로 계산한다. A3을 넣으면 A1의 보정이 엄해지는 쪽이라 보수적이다.
- A1이 아래를 **모두** 만족하면 "개선"이고 채택한다.
  1. Δ = (A1 seed 평균) − (대조군 3 seed 평균) ≥ **0.02**
  2. Δ > 대조군 seed 범위(max − min)
  3. 대응 검정(대조군 s42 vs A1_s42) 95% CI 하한 > 0, Holm 보정 McNemar p < 0.05
  4. 법령 평균 R@5가 대조군보다 표준오차 이상 떨어지지 않고, 법령별 McNemar(Holm)로 유의하게 나빠진 법령이 0개
  5. dev-b Δ(seed 평균) > 0
  6. **(추가, 더 엄격)** A1_s42와 A1_s43이 **각각** 대조군 seed 최댓값보다 높다
- **보고 항목 (판정 조건 아님)**
  - 대응 검정을 대조군 s43 vs A1_s43, s44 vs A1_s42, s44 vs A1_s43으로도 계산한다. 셋 중 하나라도 CI 하한 ≤ 0이거나 s42 대비 Δ와 0.01 넘게 다르면 한계로 적는다(검토 m1, IR 사소 2). 명령은 `uv run ragkit paired-test runs/001_baseline_seeds/<s>/eval_dev.json runs/012_remined_negatives/<run>/eval_dev.json --judge doc --k 5 --out runs/012_remined_negatives/<run>/paired_test_<s>.json`이다.
  - dev-b Δ와 함께 dev-b 대응 검정 CI(`paired_test_dev_b.json`)를 적는다(검토 m5).
  - **보조 지표**: dev의 article R@5 · R@1 · R@10 · MRR@10을 run마다 `summary.tsv`와 결과표에 적는다(`eval_dev.json`의 doc · article 블록, 검토 IR-3). A1 seed 평균의 article R@5가 대조군 평균보다 표준오차(√(p(1−p)/1243), p≈0.77이면 약 ±0.012) 이상 떨어지면, 판정과 상관없이 "같은 조 · 위임 짝 negative의 부작용 가능성"을 한계로 적는다.
- **계산량**: A1은 학습 행 · epoch · step 수가 대조군과 같다. negative 문서 길이 때문에 생기는 시간 차이는 `summary.tsv`에 적고, 1.2배를 넘으면 결과 문서에 적는다. A3은 약 1.7배인데, 채택 대상이 아니므로 계산량 대조를 두지 않는다. A3의 해석은 "계산량과 개수가 함께 바뀐 단일 seed 탐색"으로만 한다.
- 숫자는 모두 공식 `ragkit evaluate`(doc 판정) 값이다. 간이 dev는 epoch 선택에만 쓴다. 결과는 `runs/012_remined_negatives/decision.json`에 남긴다. 조건 6과 보고 항목은 `04_results.md` 표에 적는다. dev를 쓴 사실은 `experiments/ledger.tsv`에 한 줄 남긴다.

### 해석 규칙 (결과 보기 전에 고정)

- **메커니즘 (가설 c)**: 다른 법 · 법률↔시행령 혼동이 줄어서 올랐다고 쓰려면 아래가 모두 필요하다(검토 IR-2, m2).
  - 세는 방법: dev-a 실패(top-5 밖) 중 1위 오답이 "다른 법(같은 테마 · 다른 테마)" 또는 "같은 법 다른 종류(연결 있음 · 없음)"인 질문 수를, 대조군 s42 · s43 · s44와 A1_s42 · A1_s43 각각에 대해 `error_analysis.py`와 같은 관계 분류로 센다.
  - 조건 1: **A1 두 seed의 평균이 대조군 세 seed의 최솟값보다 작아야 한다.**
  - 조건 2: dev-b에서도 같은 수를 스크립트로 집계해(원문은 열지 않음) 같은 방향이어야 한다.
  - 이 조건을 못 넘으면 "R@5는 올랐지만 겨냥한 혼동 유형이 줄었다는 근거는 없다"고 쓴다.
- **기각할 때**: Δ가 0~0.02이고 CI가 0을 걸치면 "효과 없음"이 아니라 "0.02를 검출할 검정력이 부족, 크기 Δ [CI]"로 적는다(검토 m6). dev 1,243문항에서 0.02는 순변화 약 25문항이다.
- **epoch 선택 치우침**: epoch 선택과 판정이 같은 dev(dev-b 포함)를 쓴다. 대조군도 같아서 Δ에서는 대체로 상쇄되지만, dev-b가 완전히 처음 보는 데이터는 아니라는 점을 한계로 적는다(검토 m3).
- **A3의 반복 negative**: A3은 반복이 들어간 행이 543개(6.2%)라 일부 문서가 손실에 두 번 이상 들어간다(검토 m10, IR 사소 3). A3 vs A1 차이를 해석할 때 한계로 적는다.

## 로깅

| 파일 | 내용 |
|---|---|
| `runs/012_remined_negatives/data/mine.command.sh` | dump · mine · check 명령 그대로 |
| `runs/012_remined_negatives/data/mine_stats.json`, `mined_candidates.jsonl` | 채굴 통계 · 후보 manifest. 버린 "정답보다 위" 후보와 실패 질문 사유도 여기에 |
| `runs/012_remined_negatives/data/<run>/{negatives,train,dev}.jsonl`, `check.json` | run별 학습 데이터와 점검 결과 |
| `runs/012_remined_negatives/data/judge/`, `work/judge/` | L9 판정 입력 · 키 · 출력 · 집계(종류별 · 점수비 구간별) |
| `runs/012_remined_negatives/<run>/config.yaml`, `command.sh`, `train.log`, `train_meta.json` | 설정 · 명령 · 로그 · epoch별 dev · 학습 행 · digest · 시간 (`run_queue.py`) |
| `runs/012_remined_negatives/<run>/eval_dev.json`, `eval_dev_a.json`, `eval_dev_b.json`, `paired_test.json`, `paired_test_dev_b.json` | 공식 평가와 대조군 s42 대비 검정 (`run_queue.py`) |
| `runs/012_remined_negatives/<run>/paired_test_s43.json`, `paired_test_s44.json` | 다른 대조군 seed 대비 검정 (보고용) |
| `runs/012_remined_negatives/summary.tsv` | run마다 한 줄(`run_queue.py` 열) + 보조 지표 article R@5 · R@1 · R@10 · MRR@10, step 수(`train.log`에서 옮김) |
| `runs/012_remined_negatives/analysis/mechanism.out` | 해석 규칙의 유형별 실패 수(대조군 3 + A1 2, dev-a · dev-b 집계)와 집계 스크립트 |
| `runs/012_remined_negatives/decision.json` | `decide.py` 출력 |
| `runs/012_remined_negatives/data_digest.json` | 분할 · 코퍼스 · negatives · train 사본 sha256과 행 수, 학습 뒤 digest 기대값(L10), 채굴 모델 경로와 인덱스 파일 이름, git commit, uv.lock 해시, 라이브러리 버전. **학습 전에 고정** |
| `runs/012_remined_negatives/timeline.md` | 단계 · run 시작/끝 · 다시 돌리기(원인) · 계획 변경 · 예산 누적 |
| MLflow `loop/012_remined_negatives` | `run_queue.py`와 `log_experiment.py`가 자동으로 올린다 |
| `04_results.md` 재료 | 위 판정 · 보고 항목, 해석 규칙 결과, 고친 · 새로 틀린 질문(dev-a), 배포 형태 dev |

## 예상 시간 · 비용

| 작업 | 시간 | 비고 |
|---|---|---|
| 채굴 · 점검 (dump 40초 + mine 1분 + check 1분) | 약 3분 | CPU, 시험에서 확인 |
| L9 판정 200쌍 | 서브에이전트 1개, 학습과 병렬 | 데이터를 바꾸지 않음 |
| pilot_A3 | 약 3~5분 | |
| A1_s42 · A1_s43 · A3_s42 (+ 평가 각 약 5분) | 40 + 40 + 65 = 약 2시간 25분, 평가 포함 약 2시간 40분 | 대조군 s43 · s44(각 약 40분)가 끝난 뒤 |
| 다시 돌리기 (실패 정의에 걸릴 때만) | 최대 약 1시간 45분 | |
| 디스크 | 모델 약 470MB × 4 = 약 1.9GB, 채굴 산출물 약 140MB | 지우지 않는다 |

예산 80% 알림: 다시 돌리기 없이 GPU 누적 2시간 10분(2시간 40분의 80%)에 닿았는데 A3_s42가 남아 있으면, A3_s42를 뺄지 판단해 `timeline.md`에 적는다.

## 하지 않는 것

- **A3 채택, A3_s43 · C_s42(계산량 맞춤 대조)**: 첫 seed로 골라 그 seed를 평균에 넣는 치우침(검토 M3)을 피하려고 A3은 탐색으로만 둔다. A3이 유망하면 다음 바퀴에서 seed 2개 + 계산량 대조로 처음부터 돌린다.
- **437문항 되살리기**: 질문 집합을 바꾸는 두 번째 변화라 다음 바퀴 후보로 둔다.
- **컷 바꾸기**: 위임 연결 짝 빼기, 95% 컷 변경(90 · 99%), **정답보다 위 후보 살리기**, **절대 간격 컷**(정답 점수 10~90분위 0.447~0.697이라 95% 컷의 절대 간격은 약 0.022~0.035, IR 사소 5), SimANS식 범위. 모두 기록(L9 · mine_stats)만 하고 다음 후보로 둔다.
- **가짜 negative 판정으로 데이터 고치기**(002 B): 002에서 효과가 없었다. 판정은 기록용 200쌍만 한다.
- **2라운드 채굴**, ANCE식 학습 중 다시 캐기.
- **후보 B(상황형 질문 추가)**: 007 결과를 본 뒤 다음 바퀴에 판단한다. **후보 C(학습 길게)** 단독도 하지 않는다.
- lr · 배치 · epoch 탐색. 데이터가 바뀌면 최적 lr이 바뀔 수 있지만 001 설정으로 고정하고 한계로 적는다.
- 분할 · 코퍼스 변경, 라벨 고치기, test 평가, reranker · BM25 · 다른 모델.
- 점수가 낮다는 이유로 run을 다시 돌리는 것.

## 검토 반영

검토 원문: `reviews/plan_ir_expert.md`, `reviews/plan_methodology_expert.md`

| 지적 | 반영 (또는 반영하지 않은 이유) |
|---|---|
| IR-1 (중요) 95% 컷이 "정답보다 위" 후보를 모두 버려 가설 · 메커니즘이 과장됨 | 반영. 가설 문구를 "정답보다 낮으면서 정답 점수 95% 미만"으로 고치고 "닿지 못하는 범위"(45.7%, 실패 질문 사유)를 적었다. `mine.py`가 버린 후보 수 · 질문 비율 · 관계와 실패 질문 사유를 `mine_stats.json`에 기록한다. L9에 버린 "정답보다 위" 후보 50쌍을 더했다(데이터는 안 바꿈, 총 200쌍) |
| IR-2 (중요) 해석 조건 (c)에 수치 기준이 없음 | 반영. "해석 규칙"에 A1 두 seed 평균의 (다른 법 + 법률↔시행령) 1위 실패 수가 대조군 3 seed 최솟값보다 작아야 한다는 기준을 적었다. 대조군 s43 · s44도 같은 방식으로 센다 |
| IR-3 (중요) 보조 지표가 없음 | 반영. article R@5 · R@1 · R@10 · MRR@10을 보고용으로 적는다. article R@5가 표준오차 이상 떨어지면 한계로 적는다는 기준을 미리 정했다 |
| IR 사소 1 가설에 상한(평가 쪽 22%) | 반영. "닿지 못하는 범위" 2 |
| IR 사소 2 채굴 모델 = s42라 s42 대비 검정이 유리할 수 있음 | 반영. s43 · s44 대비 검정도 보고하고, Δ가 0.01 넘게 다르면 한계로 적는다 |
| IR 사소 3 A3 대체 질문의 반복 희석 | 반영. 질문당 서로 다른 negative 수 · 반복 행 수를 기록(543행, 6.2%)하고 A3 해석의 한계로 적는다 |
| IR 사소 4 "대체는 30위 밖"이 75%만 맞음 | 반영. "대부분(75%)"으로 고쳤다 |
| IR 사소 5 절대 간격 컷 | 반영. "하지 않는 것"에 다음 후보로 적었다 |
| IR 사소 6 L9를 점수비 구간별로 | 반영. `key.jsonl`에 구간(≥ 0.92 / < 0.92)을 넣고 그 구간별로 집계한다 |
| M1 (중요) L10 digest 기대값이 반드시 실패 | 반영. 학습 뒤 기준을 `data.train.n == 9126`, `train_examples == 8689`, digest A1 2c3d054b41a1 / A3 6ce401a22a7a로 바꿨다. 다시 채굴해 sha가 바뀌면 학습 전에 기대값을 다시 계산해 `data_digest.json`에 고정한다 |
| M2 (중요) "실패"가 정의되지 않음 | 반영. 실패를 비정상 종료 · OOM · NaN · `eval_dev.json` 미생성으로 한정하고, 점수 때문에는 다시 돌리지 않는다고 적었다. 다시 돌리면 같은 seed로 처음 run을 대체하고 run당 1회까지다 |
| M3 (중요) A3 승자의 저주 | 반영, 단순한 쪽 (b)를 택했다. A3은 이번 바퀴 탐색 전용이고 채택 후보는 A1뿐이다. A3_s43 · C_s42를 없애 run이 6개에서 4개로 줄었다 |
| m1 대응 검정 기준이 s42 하나 | 반영. s43 · s44 대비 검정을 보고 항목에 넣었다 |
| m2 메커니즘을 dev-a에서만 셈 | 반영. dev-b 집계 수도 같은 방향이어야 "겨냥"이라고 쓴다 |
| m3 epoch 선택과 판정이 같은 dev | 반영. 해석 규칙에 한계로 적었다. C_s42 관련 부분은 C_s42를 없애 해당이 없다 |
| m4 C_s42는 스케줄도 바꿈 | C_s42를 없애 해당이 없다. A3을 다음 바퀴에 다시 할 때 계산량 대조 설계에 반영한다 |
| m5 dev-b 조건의 검정력 | 반영. dev-b 대응 검정 CI를 판정표에 같이 적는다 |
| m6 합격선 = 기대 크기 아래쪽 | 반영. 기각 시 "검정력 부족, 크기 Δ [CI]"로 적는다는 규칙을 넣었다 |
| m7 sha가 다를 때 쓸 쪽 | 반영. 실험 폴더에서 다시 만든 파일을 쓰고, sha와 digest 기대값을 학습 전에 `data_digest.json`에 고정한다 |
| m8 `run_queue.py` 실패 동작 서술 | 반영. "실패하면 대기열 전체를 멈춘다"로 고쳤다. s43 · s44 프로세스가 옛 코드일 수 있다는 점은 `timeline.md`에 적는다 |
| m9 `01_research.md`의 옛 분할 test 숫자 | 일부 반영. 이 계획서에서는 근거로 쓰지 않는다고 제약 표에 적었다. `01` 문서 자체는 리서치 에이전트의 산출물이라 고치지 않았다. 표시를 달지는 오케스트레이터가 정한다 |
| m10 A3 반복 negative | 반영. 반복 행 수를 기록하고 한계로 적는다 |
| m11 근접 중복은 완전 일치만 봄 | 반영. 평가 쪽 정답 문서와 코사인 ≥ 0.98인 negative 수를 기록한다(시험 A1 3 · A3 4). 판정 조건으로는 쓰지 않는다 |
