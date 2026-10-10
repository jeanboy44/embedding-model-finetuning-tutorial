# 001 기본 실험: 전체 학습, 조 단위 분할

가진 질문 데이터(v1 12,232개)로 e5-small 전체를 학습한다. 2026-10-11에 법령 단위 분할로 학습한 옛 001을 지우고, 조 단위 분할로 다시 학습했다.

## 왜 조 단위인가

운영 방식은 "매뉴얼 전체로 학습하고, 매뉴얼이 추가되면 다시 학습"이다. 그래서 재야 하는 것은 처음 보는 매뉴얼(법령 단위)이 아니라 **학습한 매뉴얼 안의, 학습 질문이 없던 절**이다. 조 단위 분할은 이것을 잰다. 배경은 `archive/011_article_split/02_plan.md`.

## 재현

```bash
uv run ragkit split data/questions --unit article --test-ratio 0.15 --dev-ratio 0.1 --out data/splits_article
uv run ragkit train experiments/001_baseline/config.yaml --splits data/splits_article
uv run ragkit evaluate intfloat/multilingual-e5-small --splits data/splits_article --out experiments/results/001_baseline/000_base_test.json
uv run ragkit evaluate models/finetuned/001_baseline --splits data/splits_article --out experiments/results/001_baseline/001_test.json
uv run ragkit paired-test experiments/results/001_baseline/000_base_test.json experiments/results/001_baseline/001_test.json --judge doc
```

## 분할

같은 조의 항 · 호 조각은 한쪽으로 보내고, 법령마다 따로 나눠 54개 법령이 모든 분할에 들어간다. seed 42.

| | 질문 | 조 | 법령 | 데이터 버전 |
|---|---|---|---|---|
| train | 9,126 | 4,389 | 54 | `d9fec2ea3431` |
| dev | 1,243 | 610 | 54 | `781b8ee713a4` |
| test | 1,863 | 942 | 54 | `227edaa3e96a` |

- train hard negative 16,432개 중 dev · test 정답 조의 문서 2,072개(12.6%)를 뺐다. 그대로 두면 test 정답 조가 학습 때 오답으로 나온다.
- 그 결과 hard negative가 하나도 남지 않은 질문 437개(4.8%)는 학습 예시에서 빠졌다(학습 코드가 오답 칸이 빈 행을 건너뛴다). 학습 예시는 8,689개다.
- 법 ↔ 시행령의 비슷한 문장, test ↔ train 질문 중복 같은 누수는 아직 재지 않았다.

## 학습

설정은 `config.yaml` (CachedMNRL mini 16, 배치 32, lr 2e-5, warmup 0.1, 최대 3 epoch, 길이 512, negative 1개, seed 42, NO_DUPLICATES). M4 Pro에서 39분.

| dev (1,243개) | R@5 | MRR@10 |
|---|---|---|
| 학습 전 | 0.540 | 0.409 |
| epoch 1 | 0.697 | 0.542 |
| epoch 2 | 0.710 | 0.564 |
| **epoch 3 (선택)** | **0.717** | 0.561 |

마지막 epoch까지 R@5가 조금씩 올라 수렴했다고 보기는 어렵다.

## test (1,863개, 1회)

| | R@5 | 법령 평균 R@5 | R@1 | R@10 | MRR@10 | nDCG@10 | article R@5 |
|---|---|---|---|---|---|---|---|
| 000 학습 전 e5 | 0.545 | 0.551 | 0.284 | 0.630 | 0.392 | 0.449 | 0.627 |
| **001** | **0.744** | 0.761 | 0.449 | 0.820 | 0.573 | 0.633 | 0.797 |

| 질문 유형 | keyword (399) | question (701) | situation (763) |
|---|---|---|---|
| 000 | 0.739 | 0.573 | 0.417 |
| 001 | 0.855 | 0.760 | 0.671 |

paired-test (doc R@5): Δ **+0.199**, 95% CI [+0.177, +0.222], 고친 질문 448 / 망친 질문 77, McNemar p < 0.0001. 54개 법령 중 나빠진 법령은 없다.

- 판정: 학습 전보다 좋다(개선 기준 Δ ≥ 0.02, CI 하한 > 0, p < 0.05 모두 충족).
- 가장 많이 오른 곳은 일상 상황 질문(0.417 → 0.671)이다.
- 이 숫자는 법령 단위 test(옛 001 0.669, 학습 전 0.513)와 test가 달라 직접 비교하지 않는다.

모델: `models/finetuned/001_baseline/` (학습 기록 `train_meta.json`). 질문별 결과는 `experiments/results/001_baseline/`(커밋하지 않음).
