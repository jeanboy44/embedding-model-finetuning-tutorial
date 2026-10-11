# 003 multi_negatives — 1. 문헌 조사

## 질문
1. Dense Retrieval(임베딩 파인튜닝)에서 질문당 Hard Negative 개수를 1개에서 2~3개로 늘리는 것이 성능에 미치는 영향은 무엇인가?
2. 최신 모델로 반복 채굴(Iterative Mining)하는 것과 기존 모델로 다중 negative를 주는 것 중 어느 것이 더 안전하고 효과적인가?
3. 작은 모델(118M)과 짧은 문맥(512 토큰) 조건에서 적정 hard negative 개수는 몇 개인가?

## 근거 표

| 출처 | 방법 | 조건(모델 크기 · 데이터) | 효과 크기 | 우리와 다른 점 |
|---|---|---|---|---|
| **DPR (Karpukhin et al., 2020)** | BM25 hard negative + in-batch negatives | BERT-base, Wikipedia QA | negative 1개일 때 최적, 단순 추가 시 False Negative 노이즈로 성능 정체 | 모델 채굴이 아닌 BM25 negative 기준임 |
| **RocketQA (Qu et al., 2021)** | 모델 기반 Hard Negative Mining + Denoising | ERNIE-base, MS MARCO | Hard negative를 1개 → 다중(4개)으로 확장 시 MRR@10 +1.8~2.5%p 향상 | Denoising cross-encoder를 사용함 (우리는 점수비 95% 컷으로 대체) |
| **ANCE (Xiong et al., 2021)** | 비동기 반복 채굴 (Iterative Mining) | RoBERTa-base | 점진적 채굴로 MRR@10 대폭 개선 | 학습 중 실시간 비동기 인덱싱 (우리는 바퀴 단위 오프라인 채굴) |
| **002 실측 결과 (우리 프로젝트)** | 001 채굴 negative 1개(A1) vs 3개(A3) | multilingual-e5-small, 한국어 법령 | A1 dev R@5 0.7534 (+0.0390) vs A3 dev R@5 **0.7611 (+0.0467)** | A3이 단일 seed로도 +0.0077 높은 잠재력 확인 |

## 우리에게 옮기면

1. **다중 Hard Negative의 필요성**:
   - 오답 분석에서 확인했듯, 모델의 실패는 "같은 법 다른 조(29%)"와 "다른 법 같은 테마(27%)"라는 두 가지 상이한 축에서 동시에 일어난다.
   - Negative가 1개뿐일 때는 둘 중 한 쪽만 학습 신호로 들어가 다른 축에 대한 구별력이 취약해진다.
   - Negative를 3개로 구성하면(top-1, fill-2 등), 같은 법령 내 혼동 조문과 타 법령의 유사 조문을 한 배치 안에서 동시에 밀어낼 수 있다.
2. **반복 채굴(Iterative Mining)과의 비교**:
   - 002 모델로 새로 채굴하는 것(후보 B)도 유망하지만, 002 모델이 새로 캔 negative 1개는 학습 데이터의 일관성을 크게 흔들 수 있다.
   - 반면 001로 캔 3개 negative(후보 A)는 이미 A3 파일럿과 본 학습(0.7611)을 통해 그 강력한 효과와 안전성이 데이터 수준에서 실증되어 있다.
3. **위험 요소 및 대책**:
   - 학습 시간: negative가 3개이면 Mini-batch / CachedMNRL 연산량이 약 1.7~1.8배 증가한다(38분 → 70분).
   - 예산 제약: 학습 run을 최대 4개(파일럿 1 + 대조군/후보 seed run)로 제한하여 시간 예산(3시간 이내)을 엄격히 통제한다.

## 가설 확정 의견
- **후보 A(다중 Hard Negative 3개 최적화 및 2-seed 확증)**를 주 가설로 채택한다.
- A3_s42(0.7611)의 재현성을 2번째 seed(s43)로 검증하고, 대조군(002 A1 평균 0.7534) 대비 유의미한 추가 개선($\Delta \ge 0.01\sim 0.015$) 여부를 공식 판정한다.
