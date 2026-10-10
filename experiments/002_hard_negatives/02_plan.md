# 001 hard negative 재채굴 — 2. 실험 계획 (v2, 검토 반영)

## 가설

같은 법 1~2개뿐인 지금의 hard negative를, **모델이 직접 채굴한 상위권 negative 3개**(다른 법령 포함)로 바꾸면 dev 복수 정답 R@5가 오른다.
근거: 1위 오답의 55%가 다른 법령인데, 지금 negative는 100% 같은 법이다. 기존 negative의 40%는 이미 50위 밖이라 학습 신호가 없다 (00_error_analysis.md).

## 고정한 것 (사전 등록)

| 항목 | 값 | 근거 |
|---|---|---|
| 시작 모델 | base multilingual-e5-small (모든 run 공통) | 공정한 비교 |
| 채굴 모델 | exp_009 (best epoch 4). 채굴에만 쓴다 | 제약: e5-small만 |
| 후보 풀 | **train 법령 문서만**. dev·test 법령 문서(상위 후보의 15%)는 제외 | 처음 보는 법령 조건 유지 |
| 후보 범위 | 정답과 복수 정답을 제외한 1~10위. 3개를 못 채우면 같은 풀의 11~30위로 채우고, 그래도 모자라면 있는 것을 반복한다. 반복한 경우는 `fill`로 기록 | NV-Retriever top-10 / IR 검토 |
| negative 수 | 3개, 점수 순 | DPR, 시간 약 2배 |
| 가짜 negative 의심 | 정답 점수의 95% 이상 | NV-Retriever TopK-PercPos |
| 학습 | CachedMNRL mini 16, 배치 32, lr 3e-5, 최대 epoch 4, attention dropout 0, epoch 선택은 dev 복수 정답 R@5(간이) | 실험 009 |
| 기존 LLM hard negative | 대조군에서만 쓴다 | 40%가 50위 밖 |

## run 구성

| run | 내용 | seed |
|---|---|---|
| **대조군** | 기존 LLM negative 1개. 새 코드, 최대 epoch 4로 다시 학습한다. exp_009는 epoch 6 스케줄이라 대조군으로 쓰지 않는다 | 42, 43, 44 |
| **A** | 1~10위 중 정답 점수 95% 이상은 판정 없이 버리고 나머지에서 점수 순 3개 (NV-Retriever 방식) | 42 |
| **B** | 상위 3개 중 95% 이상 후보를 **전부 판정**한다(9,005쌍, 4,360문항). full은 (질문, full 문서, 같은 negative 3개) 학습 행을 추가하고 negative에서 뺀다. partial은 negative에서 빼기만 한다. no는 그대로 negative로 쓴다. 뺀 자리는 95% 미만 후보 중 다음 순위로 채운다 | 42 |
| **C** | A와 B 중 dev 복수 정답 R@5가 높은 쪽과 같은 설정에 **negative 1개**. 출처(채굴)와 개수(1→3)의 효과를 분리한다 | 42 |

판정 분기(보정 표본으로 판정 여부 결정)는 없앴다. B의 정의가 사전 등록대로 고정되고, 판정이 학습 run 하나보다 싸기 때문이다 (IR·방법론 검토). 2라운드 채굴은 다음 실험으로 넘긴다.

## 판정 (B, train 법령 쌍만)

- 대상: 상위 3개 중 95% 이상인 후보 9,005쌍
- 프롬프트에는 모델 점수와 순위, 어느 후보가 몇 위인지를 넣지 않고 후보 순서를 섞는다. 기준은 0단계와 같다(full / partial / no).
- 품질 확인:
  - 50쌍은 다른 agent가 다시 판정해 일치율과 κ를 잰다. κ < 0.4이면 기준을 다시 검토한다.
  - 95% 미만 후보 100쌍(대조 표본)도 판정해 가짜 negative 비율을 잰다. 5% 이상이면 결과 해석에 적는다.
- 기록: 판정 agent의 모델, 프롬프트 해시, 입력·출력 파일

## 개선 판정 기준

확증 비교는 {A, B, C} 각각 대 대조군(seed 42)의 3개로 정하고, McNemar exact p값에 Holm 보정을 적용한다. B 대 A는 탐색 비교로 보고만 한다.

"개선"은 다음을 모두 충족할 때다.
1. 대조군 3개 seed 평균 대비 dev 복수 정답 R@5의 Δ ≥ 0.02, 그리고 Δ > 대조군 seed 간 범위(max−min)
2. 대조군(seed 42) 대비 Holm 보정 McNemar p < 0.05, 그리고 질문 단위 대응 bootstrap 95% CI 하한 > 0
3. 법령 평균 R@5가 나빠지지 않고, 9개 법령 중 6개 이상이 나빠지지 않음

참고(판정 기준 아님): 법령 단위 표준오차는 ±0.040이라 처음 보는 법령 전반으로 퍼지는지는 3번으로만 본다.

## 로깅

| 파일 | 내용 |
|---|---|
| `03_run/<run>/config.yaml`, `train.log`, `train_meta.json` | 학습 설정, epoch별 dev·train 지표, 학습 시간, 학습 행 수, 데이터 해시, best epoch가 마지막 epoch인지 |
| `03_run/<run>/eval_dev.json` | 공식 평가 (질문별 qid, 법령, 순위, top-10) |
| `03_run/<run>/paired_test.json` | 대조군(seed 42) 대비 대응 검정, 보정 전후 p값 |
| `03_run/data_digest.json` | split·라벨·코퍼스·채굴 파일·채굴 모델의 sha256과 행 수, git commit, 라이브러리 버전 |
| `artifacts/mined_<run>.jsonl` | 채굴 manifest: qid, 후보 id, 순위, 점수, 정답 대비 점수비, 관계 6유형, 판정, 선택 여부, 선택 사유(top / fill / repeat) |
| `artifacts/judge/` | 판정 입력·출력, 이중 판정 일치율 |
| `04_results.md` 재료 | run별 negative 관계 분포, 학습 행 수, step 시간·최대 메모리. dump_topk + error_analysis를 다시 돌린 실패 유형표. 질문별로 고친 것과 새로 틀린 것. 학습 후 다시 채굴한 hardness |

## 예상 시간 (추정)

| 작업 | 시간 |
|---|---|
| 대조군 3 seed (negative 1개, epoch 4) | 약 3시간 (진행 중, GPU) |
| 채굴 manifest와 판정 9,005쌍 | 2~3시간 (agent 병렬, 대조군과 동시) |
| A, B, C (negative 3개 / 1개) | 약 5~6시간 (GPU 순차) |

## 하지 않는 것

손실 함수 변경(3단계), 합성 질문(4단계), 코퍼스 헤더(5단계), test 평가, 2라운드 채굴

## 검토 반영

검토 원문: `reviews/plan_ir_expert.md`, `reviews/plan_methodology_expert.md`

| 지적 | 반영 |
|---|---|
| 5% 기준을 점 추정으로 판정하면 경계에서 결과가 뒤집힌다 (200쌍이면 CI 2.4~9%) | 판정 분기를 없애고 9,005쌍을 전부 판정한다 |
| 대조군과 A는 negative의 출처와 개수가 동시에 바뀐다 | C를 "더 나은 쪽 + negative 1개"로 고정해 두 효과를 분리한다 |
| exp_009 대조군의 혼입(epoch 6 스케줄, 다른 코드) | 새 코드, 최대 epoch 4, seed 3개로 대조군을 다시 학습한다 |
| full을 negative에서 빼기만 하면 정보를 버린다 (RLHN: 다시 라벨을 붙이는 것이 가장 좋음) | full 문서로 학습 행을 추가한다 |
| 3개를 못 채우는 663문항을 반복으로 채우면 같은 negative가 겹친다 | 11~30위로 먼저 채우고, 반복은 `fill`로 기록한다 |
| Holm과 CI 조건이 섞여 계산할 수 없다 | 확증 family 3개, McNemar p에 Holm, CI는 별도 조건으로 정리했다 |
| 로깅 누락 (관계 분포, 학습 행 수, 판정 해시 등) | 로깅 표에 추가했다 |
