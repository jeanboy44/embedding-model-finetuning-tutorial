# 012 (이름 미정) — 1. 문헌

앞 단계 문서: `00_error_analysis.md`. 이 문서의 우리 쪽 숫자는 모두 00 문서와 이전 실험 문서에서 옮겼다. dev 질문별 결과 파일은 열지 않았다.
문헌 수치는 2026-10-11에 원문(arXiv HTML · PDF 본문)에서 다시 확인했다. 원문 추출 텍스트는 `runs/002_hard_negatives/work/research/pdf/*.txt`에 있다. 원문을 확인하지 못한 수치는 **"확인 못 함"**으로 표시했다.

## 질문

00 문서의 가설 후보 세 개를 문헌과 이전 실험에 비춰 본다.

| 후보 | 물음 |
|---|---|
| A | 001이 직접 캔 hard negative(train 조 문서만, 다른 법 포함, 정답 점수 95% 이상은 버림)로 기존 LLM negative를 바꾸면 dev R@5가 오르는가? 오른다면 얼마나? |
| B | train 조 조문으로 상황형(situation) 질문을 더 만들어 학습에 넣으면, 단어 겹침이 작은 질문과 situation 질문의 R@5가 오르는가? |
| C | 같은 데이터로 학습을 더 길게 · 세게(epoch 3 → 5, lr 2e-5 → 3e-5) 하면 오르는가? |

공통 조건(우리 상황): multilingual-e5-small(118M, 384차원) 하나, 한국어 법령 코퍼스, Claude가 만든 질문, 조 단위 분할, train 8,689행(negative 0개 437문항 제외), CachedMNRL 배치 32, 001 학습 39분(`runs/001_baseline_seeds/summary.tsv`, 2,369초). reranker · BM25 · LLM 확장 · 다른 임베딩 모델 금지. 새 데이터 재료는 train 조에서만.

## 근거 표

### A. 모델이 직접 캔 hard negative (가짜 negative 거르기 포함)

| 출처 | 방법 | 조건(모델 크기 · 데이터) | 효과 크기 | 우리와 다른 점 |
|---|---|---|---|---|
| NV-Retriever (arXiv:2407.15831) Table 3 | 채굴 후 정답 점수 기준 거르기(TopK-PercPos 95%: 정답 점수의 95% 넘는 후보 버림) vs 거르지 않은 top-k | e5-large-unsupervised(334M) 학습, 채굴은 **e5-mistral-7b**, negative 4개, BEIR 일부 | NDCG@10 평균 Naive top-k 0.5407 → PercPos 95% 0.5856 (**+0.045**). top-10에서 4개 샘플링도 0.5856 | 비교 대상이 "거르지 않은 top-k"라 "거르기"의 효과이지 "모델 채굴 vs LLM이 고른 negative"의 효과가 아니다. 채굴 모델이 7B로 훨씬 강하다. 우리는 001(118M) 자신으로 캔다 |
| NV-Retriever Table 1 · 4 · 5 | 채굴 모델 · 학습 모델 바꾸기 | Table 1: 같은 base, 채굴 모델만 바꿈. Table 4: Mistral-7B, negative 1개 | random negative 0.5248, BM25 negative 0.5002(Table 1). Mistral-7B·negative 1개에서도 Naive 0.6214 → PercPos 0.6479(+0.027) | negative 1개만으로도 거르기 효과가 남는다는 점은 우리 "1개만" 변형의 근거가 된다 |
| NV-Retriever Fig. 3 | LLM 판정으로 채굴 negative의 가짜 비율 측정 | Llama 3.1 70B 판정 | 거르지 않은 top-k의 가짜 negative 비율 NQ 38.8%, StackExchange 47%. 정답 기준 거르기로 57% · 50% 줄어듦(상대) | 우리 dev 1위 오답은 85%가 진짜 오답(00 라벨 표본 40개)이라 가짜 비율이 이보다 낮을 것이다. 단, train은 dev보다 모델이 잘 맞춰져 있어 상위 후보가 다를 수 있다 |
| RocketQA (arXiv:2010.08191) Table 3 · 7 | dense retriever 상위 결과를 hard negative로 쓰되 cross-encoder로 거름(점수 0.1 미만만 negative) | ERNIE base 계열 dual encoder, MS MARCO · NQ | MS MARCO MRR@10: in-batch 32.39 → **거르지 않은 hard negative 26.03(하락)** → 거른 hard negative 36.38. NQ R@5: 68.5 → 68.0 → 73.2. 사람이 본 상위 미라벨 문서 100문항 중 약 70%가 실제 정답 또는 매우 관련 | 라벨이 희소한 MS MARCO 수치다. 우리는 cross-encoder를 쓸 수 없다(제약). 대신 정답 점수 95% 컷으로 거른다 |
| ANCE (arXiv:2007.00808) Table 1 | 학습 중인 모델 자신의 ANN 인덱스에서 negative를 주기적으로 다시 캠 | BERT-base(FirstP), MS MARCO passage | dev MRR@10: Rand Neg 0.261, BM25 Neg 0.299, DPR(BM25+Rand) 0.311, ANCE 0.330 (BM25 Neg 대비 +0.031) | 기준이 BM25 negative라 우리 "LLM이 고른 같은 법 negative" 기준과 다르다. 우리는 한 라운드 정적 채굴이다 |
| DPR (arXiv:2004.04906) Table 3 | in-batch + BM25 hard negative 1개 vs 2개 | BERT-base, NQ dev | 배치 32 in-batch만(Gold 31): top-5 52.1 / top-20 70.8 → +BM25 1개 65.0 / 77.3 → +BM25 2개 64.5 / 76.4. "2개째는 더 도움이 안 된다" | negative 개수를 늘리는 효과는 작을 수 있다는 근거. BM25 negative라 우리 채굴 negative와 성격이 다르다 |
| Arctic-Embed (arXiv:2405.05374) §3.6 · §7.2 | 기존 임베딩 모델로 상위 100개를 캐고 "최대 관련도 상한"만 둬서 거름 | arctic-embed-xs~l (23M~334M; s는 e5-unsupervised-small 33M), 미세조정 단계 negative 10개 | "상한이 너무 낮거나 너무 높으면(너무 어렵거나 너무 쉬운 negative) 성능이 크게 나빠진다"(Fig. 8). 수치는 본문에 없음 → **효과 크기 확인 못 함** | 작은 모델에서도 negative 난이도 상한이 중요하다는 직접 근거. 우리 95% 컷이 그 상한 역할 |
| GPL (arXiv:2112.07577) Table · §5 | 합성 질문 학습(QGen)에 채굴 hard negative를 넣음 | MS MARCO로 학습된 dense model(DistilBERT 계열), BEIR 6개 도메인 | QGen 평균 nDCG@10 48.8 → hard negative 넣으면 **46.3(−2.5)**. "QGen은 가짜 negative에 민감하다" | 합성 질문 + 거르지 않은 채굴 negative 조합이 손해였다는 근거. B와 A를 같이 하면 특히 주의 |
| RLHN (arXiv:2505.16967) | LLM 판정 단계로 가짜 negative를 빼거나 정답으로 다시 붙임 | E5-base, Qwen2.5-7B, BGE 학습 데이터 | BEIR nDCG@10 +0.7~1.4, AIR-Bench +1.7~1.8 (초록 · HF 페이지 기준, **본문 표는 확인 못 함**) | 판정 비용이 크다. 우리 002에서 판정(B)은 거르기만 한 A보다 낫지 않았다 |
| SimANS (arXiv:2210.11773) | 정답 점수 근처의 후보를 더 자주 뽑음 | NQ, ANCE 기반 | 002 리서치가 옮긴 수치(NQ R@5 71.8 → 74.3)는 **이번에 원문 확인 못 함** | 범위를 "상위 10위" 대신 "정답 점수 근처"로 고르는 변형의 근거일 뿐 |
| **우리 002 A (옛 법령 단위 분할)** `experiments/archive/002_hard_negatives/04_results.md` | 같은 방식: 학습된 모델로 train 법령 문서만 캐고, 정답 점수 95% 이상 버림, 상위 10위에서 3개 | e5-small, 학습 행 8,213, 대조군 seed 3 | dev 복수 정답 R@5 0.720 → **0.761 (+0.041, CI [+0.023, +0.056])**. 1개만(C) +0.031 [+0.013, +0.046]. 3개 vs 1개 +0.010, CI [−0.003, +0.023] 유의하지 않음. 시간 55분 → 95분(1.7배). 판정까지 한 B는 +0.036으로 A와 차이 없음 | 분할이 법령 단위였고 판정이 복수 정답 기준이었다. 이번은 조 단위 · doc 기준. 채굴 모델도 003(epoch 6)이었고 이번은 001(epoch 3) |

### B. LLM 합성 질문 추가

| 출처 | 방법 | 조건(모델 크기 · 데이터) | 효과 크기 | 우리와 다른 점 |
|---|---|---|---|---|
| Promptagator (arXiv:2209.11755) | 몇 개(0~8) 예시로 LLM(FLAN)이 문서별 질문 생성 → 생성 데이터로 학습한 retriever가 자기 출처 문서를 top-K에 못 올리면 그 질문을 버림(round-trip 거르기) | **110M** T5-base dual encoder, BEIR 11개, 사람 라벨 없이 | ColBERT v2보다 평균 1.2 nDCG 이상 높음(초록). round-trip 거르기: 11개 중 8개에서 오르고 **평균 +2.5**. 생성 질문이 적은 작은 데이터셋(NFCorpus · SciFact)에선 거르기가 손해 | 우리는 이미 사람 라벨 대신 생성 질문 9,126개로 학습 중이다. 0에서 시작하는 효과가 아니라 "이미 있는 생성 질문에 더하기"의 효과는 문헌에 없다. 거르기가 "좋은 어려운 질문"도 버린다고 저자가 적었다 |
| GPL (arXiv:2112.07577) | 생성 질문(QGen) + cross-encoder 점수 증류 | DistilBERT 계열, BEIR 6개 | MS MARCO 모델 45.2 → QGen 48.8(+3.6) → GPL 51.5. QGen + 채굴 negative는 46.3 | 도메인 적응(라벨 0) 조건. cross-encoder 증류는 우리 제약상 못 쓴다 |
| InPars (arXiv:2202.05144) | GPT-3 few-shot으로 합성 질문 | — | 초록에 수치 없음 → **효과 크기 확인 못 함** | 참고만 |
| Doc2Query-- (arXiv:2301.03266) | 생성 질문 중 관련도 모델 점수가 낮은 것(환각)을 버림 | 문서 확장(sparse) 설정 | 거르기로 검색 효과 최대 +16%, 색인 33% 감소 | 생성 질문 환각이 실제로 해롭다는 근거. 학습 데이터가 아니라 색인 확장이다 |
| Arctic-Embed §3.5 | 합성 질문 생성 시 negative 문서도 함께 보여 줘 질문을 구체화 | arctic-embed 계열 | 두 합성 데이터셋이 원래 HotpotQA 학습에 "가까운" 향상(Fig. 4, 수치 본문에 없음 → **확인 못 함**) | 생성 프롬프트에 헷갈리는 이웃 조문을 같이 보여 주면 질문이 정답 조만 가리키게 된다는 실무 팁 |
| Roque 외, PACLIC 2024 (aclanthology 2024.paclic-1.35) | 필리핀 세법(NIRC) 1,020개 사람 질문을 본떠 Llama 3.1 8B로 합성 질문 7,310개, 임베딩 미세조정 | GTE · Nomic · Jina 등, 법령 검색 | **조문을 작은 단위로 나눈 것이 Top-1 최대 +13%로 가장 컸고**, 합성 질문 미세조정은 나눈 조문 위에서 "2% 미만, 거의 개선 없음" | 법령 · 합성 질문이라 가장 가깝다. 생성 모델이 8B로 약하다. 우리는 이미 항 · 호 조각으로 나눠져 있다 |
| STARD (arXiv:2406.15313) | 비전문가의 실제 법률 상담 질문 1,543개, 조문 55,348개 | 중국 법령 검색 | 최고 방법도 Recall@100 0.907. "일반인 질문은 정확한 법률 용어와 인용이 없다" | 우리 situation 질문의 어려움(겹침 낮음)과 같은 현상. 실제 사용자 질문이고 우리는 생성 질문이다 |
| **우리 007 구어 질문 (옛 분할, 진행 중)** `.claude/worktrees/r003-colloquial/runs/003_colloquial/03_run/A_s42/paired_dev*.txt` | train 원 질문을 줄임말 · 은어 · 일상어로 다시 쓴 변형 1,898개 추가(정답 · negative는 원 질문 그대로) | e5-small, 대조군 = 002 A, seed 42 하나만 끝남 | 원 dev 복수 정답 R@5 0.761 → 0.756 (**−0.005**, CI [−0.018, +0.008]). 구어 dev 0.492 → 0.511 (+0.020, CI [−0.011, +0.053], p 0.31). 학습 2시간 7분 | 새 조문을 넣은 게 아니라 같은 정답의 말 바꾸기다. 옛 분할 · 단일 seed라 결론이 아니다. 그래도 "원 dev는 안 오른다"는 방향은 B에 불리한 정황 |

### C. 학습을 더 길게 · 세게

| 출처 | 방법 | 조건 | 효과 크기 | 우리와 다른 점 |
|---|---|---|---|---|
| **우리 003 수렴 확인 (옛 분할)** `experiments/README.md` | 001과 같고 epoch 6, lr 3e-5 | e5-small | dev 복수 정답 R@5 0.712 → 0.721 (**+0.009**, best epoch 4) | 분할이 다르다. 단일 run |
| **우리 002 대조군 epoch별** `experiments/archive/002_hard_negatives/04_results.md` | 기존 negative, 4 epoch, seed 3 평균 | e5-small | epoch 1→4: 0.693, 0.700, 0.708, 0.709. epoch 3 → 4 +0.001 | 기존 LLM negative로는 epoch 3~4에서 평평해졌다 |
| 001 (조 단위) epoch별 `runs/001_baseline_seeds/s42/train_meta.json` (00 문서) | 3 epoch | e5-small | 0.697 → 0.710 → 0.717 (마지막 epoch +0.007) | 아직 오르지만 폭이 줄고 있다 |
| 005 배치 128 · 004 LoRA (옛 분할) `experiments/README.md` | 학습 설정만 바꿈 | e5-small | 001(배치 32)보다 test가 낮았다(0.659, 0.641) | 설정 변경 계열이 일관되게 이득이 작았다 |
| 문헌 | 같은 데이터로 epoch만 늘린 효과를 다룬 retrieval 논문은 찾지 못했다. Arctic-Embed도 미세조정 epoch 수를 적지 않았다 | — | **근거 없음** | — |

## 알려진 함정

| 함정 | 어디서 | 우리에게 |
|---|---|---|
| 가짜 negative(사실 정답인 후보를 negative로) | RocketQA(거르지 않으면 MRR 32.4 → 26.0), NV-Retriever(top-k의 39~47%), GPL(QGen+HN −2.5) | 정답 점수 95% 컷 필수. 특히 **위임 연결된 법률↔시행령 짝**은 00 라벨 표본 partial 5건 중 2건이었다(상위 규범이 근거만, 하위가 세부). 캔 negative 중 위임 연결 짝 비율을 세어 기록하자 |
| 너무 어려운 negative | Arctic-Embed §7.2, 002 B(가장 어려운 진짜 negative를 쓴 B가 train은 더 맞추고 dev는 못 넘음) | 95% 컷을 느슨하게(예: 99%) 하지 말 것. 상위 1~2위만 고집하지 말고 상위 10위 안에서 고른다 |
| 너무 쉬운 negative | 00: 기존 HN의 39%가 001 기준 50위 밖 | A가 고치려는 바로 그 문제 |
| 평가 쪽 문서 누수 | 00: 001로 캔 후보의 20%가 dev · test 정답 조 문서. 002에서 15~19% | 후보 풀을 train 조 문서로 제한(제약 2절). 단, 이 때문에 **dev 실패의 22%를 차지하는 "평가 쪽 조 문서가 1위" 혼동은 직접 negative로 줄 수 없다.** 같은 모양의 train 쪽 혼동에서 일반화되기를 기대하는 것이다 |
| 채굴 모델이 train에 맞춰져 있어 채굴이 쉬워 보임 | 002 리서치 검토(train 정답 1위 0.61 vs dev 0.45) | 001은 train R@5 0.808로 덜 맞춰져 있어 이번엔 덜하다. 정답 점수가 낮은 질문은 95% 컷에 후보가 다 걸려 negative 0개가 될 수 있다(002 81문항, 007 구어 20%) → 처리 규칙을 계획서에 미리 |
| 합성 질문의 분포 차이 · 환각 | Doc2Query--, Promptagator(거르기가 좋은 어려운 질문도 버림), PACLIC(합성 질문 미세조정 <2%) | 우리 dev도 **같은 스킬로 생성한 질문**이다. B가 dev에서 오르더라도 "생성기 말투에 맞춘 효과"일 수 있고 실제 사용자 질문으로 옮겨 간다는 보장은 없다. 거꾸로 dev와 말투가 같아 B가 dev에선 유리하게 보일 수 있다 |
| 질문 양이 성능을 가르지 않는다는 우리 정황 | 00: 법령별 R@5와 train 질문 수 순위상관 0.01, 조 커버리지 0.08 | B("질문을 더")에 불리한 내부 근거 |
| 계산량 차이 | 002: negative 3개면 1.7배 | negative 1개 변형은 대조군과 계산량이 같아 깨끗하다 |

## 우리에게 옮기면

**A (가장 근거가 강함)**
- 같은 방식이 옛 분할에서 +0.041(1개만 +0.031)이었고, 문헌의 방향(채굴 + 정답 기준 거르기 > 거르지 않음 / BM25 · 무작위)과 맞는다. 00 분석에서도 현재 HN이 실패와 다른 곳(같은 조 조각 50%, 다른 법 0%)을 겨냥하고, 001로 캔 후보의 관계 분포가 dev 실패 분포와 거의 같다.
- 예상 크기: **+0.02~0.04**로 본다. 근거는 002의 +0.031~0.041이다. 낮춰 잡는 이유는 세 가지다. (1) 채굴 모델 001이 002 때의 003보다 약하다(epoch 3). (2) 조 단위에서는 dev 실패 1위 문서의 22%가 평가 쪽 조라 negative로 못 준다. (3) 판정 기준이 doc(정답 하나)이라 복수 정답 판정보다 움직임이 작을 수 있다. 문헌 숫자(NV-Retriever +0.045)는 "거르지 않은 top-k 대비"이고 채굴 모델이 7B라 그대로 기대하지 않는다.
- 설정은 002 A의 사전 등록값(상위 10위, 95% 컷, 판정 없음)을 그대로 쓰는 게 근거가 있다. negative 개수는 DPR(2개째 이득 없음)과 002(3 vs 1 차이 유의하지 않음)를 보면 **1개 변형이 계산량을 대조군과 맞춘 채 출처 효과만 재는 깨끗한 run**이다. 3개 변형은 시간 1.7배(약 65분)에 추가 이득이 불확실하다.
- negative 0개가 된 437문항(학습에서 빠졌던 것)은 채굴로 되살아날 수 있다. 다시 0개가 되는 질문 수와 처리 규칙(기존 LLM negative로 대체 등)을 계획서에 미리 적어야 한다(lessons.md).
- 판정(300쌍 시범)은 002에서 결과를 바꾸지 못했으므로 필수가 아니다. 다만 위임 연결 짝 · 같은 조 다른 항처럼 partial이 몰리는 관계가 negative에 얼마나 들어가는지는 세어 기록할 가치가 있다.

**B (근거가 약하고 비용이 큼)**
- 문헌의 합성 질문 이득(Promptagator, GPL, QGen +3.6)은 거의 다 **사람 라벨이 없던 도메인에 0에서 더한 효과**다. 우리는 이미 생성 질문으로 학습 중이라 "더 넣기"의 한계 효과는 문헌에 근거가 없다. 법령 쪽 가장 가까운 연구(PACLIC 2024)는 합성 질문 미세조정이 2% 미만이었다.
- 내부 정황도 불리하다: 법령별 R@5가 train 질문 수와 무관(0.01), 007 구어 변형은 원 dev −0.005(단일 seed, 옛 분할).
- 그래도 B는 A가 못 닿는 범위(정답 31위 밖 41%, 겹침 Q1 R@5 0.490)를 겨냥하는 유일한 후보다. 002 회고도 "hard negative만으로는 크게 놓친 실패를 못 고친다"고 적었다. 한다면 Promptagator식 거르기(001이 자기 출처 조문을 top-K에 올리는지)를 쓰되 K를 넉넉히 해 어려운 상황 질문을 지나치게 버리지 않도록, Arctic식으로 생성 프롬프트에 헷갈리는 이웃 train 조문을 같이 보여 주는 게 근거 있는 방법이다.
- 비용: 2,000~4,000문항 생성 + 근사 중복 검사 + 시범 300문항 품질 확인. 학습 행이 늘어 run당 50~60분. 007 결과(다른 worktree)가 끝나기 전에 시작하면 중복 투자가 될 수 있다.

**C (단독 효과는 합격선 아래로 예상)**
- 문헌 근거는 없다. 내부 근거는 003 +0.009, 002 대조군 epoch 3 → 4 +0.001, 001 마지막 epoch +0.007이다. 합격선 Δ ≥ 0.02를 넘을 가능성이 낮다.
- 쓸모는 **A 3개 변형의 계산량 맞춤 대조**다(A가 1.7배 길면 "그냥 오래 돌려서" 오른 게 아님을 보여야 한다, 제약 3절 마지막 줄). A를 1개 변형으로 하면 계산량이 같아 C가 필요 없다.
- epoch · lr를 같이 바꾸면 두 가지가 섞이므로, 한다면 하나만 바꾼다.

## 우선순위 의견

| 순위 | 후보 | 근거 | 이번 바퀴 비용(run 8개 이하, 대조군은 공용 `runs/001_baseline_seeds` 재사용으로 0개) |
|---|---|---|---|
| 1 | **A** — 001로 캔 negative 1개(계산량 = 대조군) + 3개 변형 | 문헌 방향 일치, 옛 분할 같은 방법 +0.031~0.041, 00의 겨냥 어긋남 숫자가 가장 직접적 | 파일럿 1 + A1(약 40분) + A3(약 65분) + 채택 확인 seed 1(40~65분) = 본 run 3개, 약 2.5~3시간. 채굴은 CPU 수 분, 판정 없음 |
| 2 | **C** — A3을 채택할 때만 계산량 맞춤 대조로 1 run | 단독 기대 Δ < 0.02, 합격선 아래 | 1 run(약 65분). A1을 채택하면 생략 |
| 3 | **B** — 이번 바퀴에서는 미루고 다음 바퀴 후보로 | A가 못 닿는 31위 밖 실패를 겨냥하는 유일한 수단이지만, "이미 생성 질문으로 학습 중인 상태에서 더하기"의 문헌 근거가 없고 내부 정황(질문 수와 R@5 무관, 007 원 dev −0.005)이 불리하며 생성 · 검사 비용이 크다. 007이 끝나면 그 결과로 다시 판단 | 생성 + 검사 + 1~2 run |

남는 예산(본 run 3~4개)은 A 안의 변형 하나(예: 위임 연결 짝을 negative에서 뺀 변형)나 채택 확인 seed 추가에 쓰는 것을 제안한다. 다만 변형을 늘릴수록 비교가 많아져 Holm 보정이 엄해진다.

## 가설 수정 여부

- **A**: 수정. 기대 효과를 "옛 분할 +0.041"이 아니라 **+0.02~0.04**로 낮춰 잡는다(채굴 모델이 더 약함, 평가 쪽 혼동 22%는 직접 못 줌, doc 판정). 출처 효과를 계산량과 분리하려면 **negative 1개 변형**을 주 run으로 두는 것을 권한다. 위임 연결 짝은 가짜 negative 위험 구간이므로 비율을 기록한다.
- **B**: 유지하되 이번 바퀴 우선순위에서 내린다. 한다면 Promptagator식 round-trip 거르기(K 넉넉히)와 이웃 조문을 보여 주는 생성 프롬프트를 쓰고, dev가 같은 생성기의 질문이라는 한계를 결과에 적는다.
- **C**: 단독 가설로는 권하지 않는다. A가 계산량을 늘릴 때의 대조로만.

## 출처

- NV-Retriever: https://arxiv.org/html/2407.15831 (Table 1, 3, 4, 5, Fig. 3 본문 확인)
- Arctic-Embed: https://arxiv.org/html/2405.05374 (§3.5, §3.6, §4.5, §7.2, Table 1, 3 본문 확인)
- RocketQA: https://arxiv.org/abs/2010.08191 (PDF Table 3, 7과 70% 서술 확인, `runs/002_hard_negatives/work/research/pdf/rocketqa.txt`)
- ANCE: https://arxiv.org/abs/2007.00808 (PDF Table 1 확인, `ance.txt`)
- DPR: https://arxiv.org/abs/2004.04906 (PDF Table 3 확인, `dpr.txt`)
- GPL: https://arxiv.org/abs/2112.07577 (PDF 결과표 · §5 확인, `gpl.txt`)
- Promptagator: https://arxiv.org/abs/2209.11755 (PDF §3.2, §4.3, 모델 표 확인, `promptagator.txt`)
- InPars: https://arxiv.org/abs/2202.05144 (초록만, 수치 없음)
- Doc2Query--: https://arxiv.org/abs/2301.03266 (초록)
- RLHN: https://arxiv.org/abs/2505.16967 (초록 · HF 페이지만, 본문 표 확인 못 함)
- GISTEmbed: https://arxiv.org/html/2402.16829 (Table 1 확인. bge-small에서 retrieval 51.68 → GIST 50.73으로 오히려 낮음. 이번 후보와 직접 관련은 적어 표에서 뺐다)
- Roque 외 PACLIC 2024: https://aclanthology.org/2024.paclic-1.35 (PDF 초록 확인, `paclic.txt`)
- STARD: https://arxiv.org/abs/2406.15313 (초록)
- SimANS: https://arxiv.org/abs/2210.11773 (이번에 원문 확인 못 함)
- 이전 실험: `experiments/archive/002_hard_negatives/{01_research,04_results,05_retro}.md`, `experiments/README.md`, 007 worktree `runs/003_colloquial/02_plan.md` · `03_run/A_s42/paired_dev*.txt`
