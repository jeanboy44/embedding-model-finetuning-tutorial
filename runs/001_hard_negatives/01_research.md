# 001 hard negative 재채굴 — 1. 리서치

이번 실험의 질문: **모델이 직접 채굴한 hard negative(다른 법령 포함)로 다시 학습하면, 처음 보는 법령(dev)에서 복수 정답 R@5가 오르는가?**
제약: e5-small만 사용(채굴과 guide도 e5-small 자신), cross-encoder 없음, 가짜 negative 판정은 Claude.

## 근거 (2026-10-01 리서치 agent 보고 + 원문 확인)

| 기법 | 보고된 효과 | 출처 | 우리에게 주는 시사점 |
|---|---|---|---|
| 정답 점수 기준 필터 (TopK-PercPos, 정답 점수의 95% 넘는 후보 제외) | e5-large-unsup, BEIR 3종 평균 nDCG@10 0.541 → 0.586 (+4.5pt). 가짜 negative 50~57% 감소 | NV-Retriever, arXiv:2407.15831 (Table 3, 5) | 95% 기준을 그대로 쓰되, 걸러진 후보는 버리지 않고 판정해 복수 정답으로 돌린다 |
| negative 개수·범위 | top-10 안에서 4개를 뽑을 때 최적. 범위를 k>10으로 넓히면 하락 | NV-Retriever Fig.2 | 상위 30개 전체가 아니라 상위권 위주로 뽑는다 |
| 더 강한 모델로 채굴 | 채굴 모델만 바꿔 +3.2pt (e5-large-unsup → e5-mistral-7b) | NV-Retriever Table 1 | 우리는 e5-small 자신만 쓸 수 있다 → 이득이 작을 수 있음 |
| 노이즈 제거 (RocketQA) | 거르지 않고 hard negative를 넣으면 MS MARCO MRR@10 32.4 → **26.0으로 하락**, 거르면 36.4. 상위 결과 중 라벨 없는 문서의 약 70%가 실제 정답 | arXiv:2010.08191 (Table 3, 7) | 가짜 negative 처리가 필수다. 우리 dev 판정에서 구조 후보의 full 비율은 1.4%였지만, 모델이 상위로 올린 후보는 훨씬 높을 수 있다 |
| 반복 채굴 (ANCE) | MS MARCO MRR@10: BM25 negative 0.299 → ANCE 0.330 | arXiv:2007.00808 | 한 라운드 효과를 보고 반복 여부를 정한다 |
| 동적 재채굴 (Conan) | CMTEB 평균 68.8 → 71.2 | arXiv:2408.15710 | 같음 |
| 모호 구간 샘플링 (SimANS) | NQ에서 ANCE R@5 71.8 → 74.3 | arXiv:2210.11773 | 정답보다 너무 높거나 너무 낮은 후보 대신 정답 근처 점수의 후보를 뽑는다 |
| 한국어 e5-small: GISTEmbed vs MNRL | nDCG@10: 원본 0.671 → MNRL **0.626(하락)**, GIST 0.678, GIST+margin 0.686 | HF blog dragonkue (비동료심사) | 한국어에서 가짜 negative 처리가 특히 중요하다는 정황 (3단계 손실 함수에서 다룬다) |
| 한국 소방 법령 | BM25 hard negative 3개로 BGE-M3 파인튜닝 R@10 53.8 → 57.3 | SearchFireSafety, arXiv:2604.06173 | 한국 법령에서도 hard negative 학습 이득이 확인됨(폭은 작음) |
| E5 원 학습 레시피 | hard negative 7개, τ=0.01(scale 100) | arXiv:2212.03533 | negative 수와 scale의 참고값 |

sentence-transformers 6.1에서 쓸 수 있는 것 (설치 코드에서 확인):
- `util.mine_hard_negatives(model, ..., range_min, range_max, relative_margin, num_negatives, sampling_strategy, output_format="n-tuple")`
- `CachedMultipleNegativesRankingLoss`의 `hardness_mode`. 3단계에서 쓴다.
- 법령 필터(같은 법령 제외 등)는 없어서 후처리로 해야 한다. 우리는 이미 `dump_topk.py`로 상위 50개를 갖고 있으므로 직접 고른다.

## 우리 에러 분석과 맞춰 본 것

- 1위 오답의 55%가 다른 법령이다. 문헌의 "코퍼스 전체에서 채굴"이 정확히 이 부분을 겨냥한다.
- 기존 hard negative의 40%는 50위 밖이라 쓸모가 없다. 문헌의 "쉬운 negative는 학습 신호가 없다"와 맞는다.
- 정답 점수의 95%를 넘는 후보가 문항당 2.7개(train 22,622개)다. RocketQA의 "상위 미라벨 문서 70%가 실제 정답" 수준이라면 가짜 negative가 대량으로 섞일 수 있다. 판정이 필요하다.

## 열린 질문 (검토 요청)

1. 95% 기준과 상위 범위(예: 1~30위)를 우리 데이터에서 어떻게 정할지. dev로 고르면 dev를 학습 설계에 쓰는 셈인데 허용 범위인가?
2. 판정량 22,622개를 줄이는 방법. 단어 겹침 필터(0단계에서 버린 쌍 중 정답 0/100), 95% 이상 중 상위 몇 개만 판정, 판정 없이 95% 이상은 그냥 버리기(NV-Retriever 방식) 중 무엇이 근거가 있는가?
3. 다른 법령과 같은 법령 negative의 비율을 정해야 하는가, 점수 순으로 두면 되는가?
4. negative 개수(1 → 5~7)를 늘리면 CachedMNRL 메모리·시간이 어떻게 되나 (배치 32 × 질문당 1+1+k 문장)
5. 반복 채굴은 이번 실험에 넣을지, 한 라운드 결과를 보고 다음 실험으로 할지

## 검토 반영 (2026-10-02)

검토 원문: `reviews/research_ir_expert.md`(dense retrieval 학습 전문가), `reviews/research_methodology_expert.md`(실험 방법론 전문가)

바로잡은 것:
- **NV-Retriever +4.5pt를 그대로 기대하면 과장이다.** 7B 모델로 채굴한 결과이고, 비교 대상도 걸러 내지 않은 top-k다. 우리는 e5-small 자신으로 채굴하므로 이득이 작을 수 있다.
- **RocketQA의 "상위 미라벨 문서 70%가 정답"은 라벨이 희소한 MS MARCO의 수치다.** 우리 dev에서 판정된 질문의 정답 점수 95% 이상 후보 3,731개 중 full은 0.6%, partial은 4.7%였다(구조 후보만 판정했으므로 하한).
- **train에서 채굴하면 실제보다 쉬워 보인다.** exp_009는 train에 맞춰져 있다(정답 1위 비율 train 0.61, dev 0.45 / 95% 이상 후보 문항당 train 2.7, dev 5.4).

새로 확인한 것:
- **법령 단위로 다시 뽑은 표준오차는 ±0.040이다.** 질문 단위 ±0.012의 3배다. 처음 보는 법령으로의 일반화는 이보다 훨씬 불확실하다. micro는 0.729, 법령 평균은 0.762다.
- **train 채굴 후보(top-30)의 19%가 dev·test 법령 문서다**(dev 7%, test 12%). 코퍼스 전체에서 채굴하면 dev·test 문서가 학습에 들어가 "처음 보는 법령" 조건이 깨진다.
- **추가 근거:**
  - LLM 판정은 사람과의 일치도가 낮다(RLHN, κ 0.32~0.39). 그래도 다시 라벨을 붙이는 것이 버리는 것보다 낫다.
  - 너무 어려운 negative와 너무 쉬운 negative 모두 성능을 떨어뜨린다(Arctic-Embed).
  - DPR에서는 hard negative 2개째부터 이득이 없었다.

반영한 결정 (02_plan.md로 넘김):
1. **채굴 후보는 train 법령 문서로 제한한다.** 코퍼스 전체 채굴은 "코퍼스 전체를 알고 학습" 조건(결정 B)으로 따로 표시할 때만 한다.
2. **후보 범위는 상위 10위, negative는 3개, 점수 순으로 고른다.** 다른 법령 비율은 강제하지 않는다. 지금도 dev 실패 분포와 비슷하다.
3. **95% 기준은 dev가 아니라 train 판정 표본 200쌍으로 보정한다.** 문헌 기본값(95%, 상위 10위, 3개)을 사전 등록하고, dev로 비교하는 변형은 3개까지로 제한한다.
4. **판정은 실제로 뽑힐 후보만 한다**(문항당 최대 3개). full은 복수 정답으로 돌리고, partial은 negative에서 빼기만 한다. 판정 프롬프트에는 모델 점수와 순위를 보여 주지 않고 후보 순서를 섞는다. 50쌍은 두 번 판정해 일치율을 잰다.
5. **개선 판정 기준**(방법론 전문가, 모두 충족해야 함):
   - Δ ≥ 0.02
   - 질문 단위 대응 bootstrap CI 하한 > 0 (McNemar 병행)
   - Δ > seed 간 차이의 2배
   - 법령 평균이 나빠지지 않고, 9개 법령 중 6개 이상이 나빠지지 않음
6. **로깅:** eval 결과의 질문별 항목에 qid와 top-10을 넣는다. 채굴 manifest, 판정 파일, 데이터 digest, 대응 검정 결과(`paired_test.json`)를 남긴다.
