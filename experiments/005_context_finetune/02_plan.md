# 005 context_finetune — 2. 실험 계획

## 가설

**학습 및 평가 코퍼스 전체에 `[{law_name} > {chapter}] {title}\n{text}` 형태의 계층적 맥락 헤더를 적용하여 모델을 파인튜닝(002 A1 설정: CachedMNRL, lr 2e-5, epoch 3, seed 42/43)하면, 004 제로샷의 학습-추론 분포 불일치가 해소되고 모델이 헤더 구조에 최적화된 어텐션을 형성하여 dev R@5가 002 대조군 평균(0.7534) 대비 0.010 이상 유의하게 상승(목표 0.765+)할 것이다.**

---

## 제약 확인 (`references/constraints.md`)

| 항목 | 이 계획에서 지키는 방법 |
|---|---|
| 1 · 모델은 e5-small 하나 | 모델 가중치·크기(118M, 384) 불변, reranker/BM25 없음 |
| 2 · 새 학습 재료는 train 조에서만 | `splits_article/train.jsonl` 질문과 train 조문만 학습에 투입 |
| 2 · 오답 분석은 dev-a에서만 | dev-a 질문만 열람, dev-b는 집계 수치만 확인 |
| 2 · test 셋 미사용 | test 분할은 건드리지 않고 dev로만 판정 |
| 3 · 대조군 seed 재사용 | `runs/002_hard_negatives`의 A1_s42(0.7538), A1_s43(0.7530)을 공용 대조군으로 재사용 (평균 0.7534) |
| 3 · 채택 후보 seed 2개 이상 | `header_s42`, `header_s43` 2개 seed 확증 |
| 4 · 학습 run 8개 이하 | 파일럿 1 + 본 학습 2 = **총 3 run** |
| 4 · 본 학습 전 파일럿 | `pilot_header` (30 step, 300문항, dev 미평가) 선행 |

---

## 고정한 것 (사전 등록)

| 항목 | 값 | 근거 |
|---|---|---|
| 학습 하이퍼파라미터 | CachedMNRL mini 16, 배치 32, lr 2e-5, warmup 0.1, 3 epoch | 002 A1과 완벽히 동일 스케줄 통제 |
| 학습 질문 집합 | 8,689행 (`data/splits_article/train.jsonl`) | 질문 집합 불변 |
| Hard Negative | `runs/002_hard_negatives/data/A1/negatives.jsonl` (1개 채굴 음성) | 음성 출처 불변 |
| 평가 셋 | `data/splits_article/dev.jsonl` (1,243문항) | 불변 |
| **바꾸는 것** | **학습 및 평가 코퍼스 전체에 맥락 헤더 적용** | `law_docs.json` → `law_docs_header.json` |

---

## run 구성

| run | 바꾸는 것 | seed | 코퍼스 | 예상 시간 | 역할 |
|---|---|---|---|---|---|
| **대조군 (002 A1)** | 기준선 | 42, 43 | 원본 코퍼스 | 0 (재사용) | 공용 대조군 (평균 0.7534) |
| `pilot_header` | 헤더 코퍼스, 30 step 파일럿 | 42 | `law_docs_header.json` | ~2분 | 파일럿 (손실/메모리 점검) |
| `header_s42` | 헤더 코퍼스 학습 1차 | 42 | `law_docs_header.json` | ~70분 | 주 가설 후보 (Seed 42) |
| `header_s43` | 헤더 코퍼스 학습 2차 | 43 | `law_docs_header.json` | ~70분 | 주 가설 후보 (Seed 43) |

---

## 개선 판정 기준 (합격선)

- 대조군: 002 A1 2-seed 평균 (**0.7534**)
- **개선 통과 기준**:
  1. $\Delta(\text{후보 seed 평균} - \text{대조군 seed 평균}) \ge 0.010$ (dev R@5 $\ge 0.7634$)
  2. $\Delta >$ 대조군 seed 범위 (0.0008)
  3. 질문 단위 대응 검정 95% CI 하한 $> 0$
  4. dev-b에서도 $\Delta > 0$ (동반 상승)
  5. 유의하게 나빠진 법령 0개

---

## 누수 점검

- L1: `law_docs_header.json` 25,967개 전수 일치 및 원형 보존 확인 완료
- L2: 004에서 검증된 8종 누수/무결성 테스트 통과 데이터 재활용
- L3: train/dev/test 분할 불변
