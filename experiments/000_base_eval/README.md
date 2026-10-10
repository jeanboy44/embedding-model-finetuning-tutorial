# 000 학습 전 모델 평가 (기준선)

학습하지 않은 `intfloat/multilingual-e5-small`을 평가셋으로 잰다. 모든 실험의 출발점이다.

## 재는 방법

```bash
uv run ragkit compare experiments/000_base_eval/config.yaml          # test 전체, 결과는 results/
uv run python lecture/05_data_eval/03_metrics.py                     # 강의 5교시: 같은 값을 ragkit evaluate로
uv run ragkit evaluate intfloat/multilingual-e5-small --split dev --labels data/labels/dev.jsonl   # dev 복수 정답
```

코퍼스 임베딩은 `data/processed/index/multilingual-e5-small.sqlite`를 재사용한다(`scripts/finetuned_drive.py download`로 받는다). 없으면 처음 한 번 7~13분 걸려 만든다.

## 값: 조 단위 분할 (2026-10-11부터 기준)

test (1,863개, `data/splits_article`, 데이터 버전 `227edaa3e96a`, 출처 `001_baseline/README.md`)

| R@5 | 법령 평균 R@5 | R@1 | R@10 | MRR@10 | nDCG@10 | article R@5 |
|---|---|---|---|---|---|---|
| **0.545** | 0.551 | 0.284 | 0.630 | 0.392 | 0.449 | 0.627 |

dev (1,243개) R@5 0.540 · MRR@10 0.409 (`models/finetuned/001_baseline/train_meta.json`의 학습 전 값)

## 값: 법령 단위 분할 (옛 기준, 강의 5~6교시 숫자)

이 폴더를 만들기 전에 이미 잰 값이다. 같은 모델 · 같은 분할이라 다시 재도 같다.

test (2,526개, 데이터 버전 `a0058b5ae8d9`, 출처 `archive/exp_010_lecture_comparison/results/comparison.md`)

| R@5 | 법령 평균 R@5 | R@1 | R@10 | MRR@10 | nDCG@10 | article R@5 |
|---|---|---|---|---|---|---|
| **0.513** | 0.540 | 0.284 | 0.601 | 0.381 | 0.434 | 0.577 |

| 질문 유형 | keyword | question | situation |
|---|---|---|---|
| R@5 | 0.725 (n=542) | 0.517 (n=911) | 0.403 (n=1073) |

dev (1,412개, 데이터 버전 `992753ff49e1`, 출처 `archive/exp_008_dev_multi_baseline/results/comparison.md`)

| R@5 | 복수 정답 R@5 | 법령 평균 R@5 | R@10 |
|---|---|---|---|
| 0.550 | 0.562 | 0.571 | 0.623 |

- 일상 상황(situation) 질문에서 가장 약하다(0.403). 학습으로 올릴 곳이다.
- `archive/exp_005_base_model_comparison`의 0.526은 질문 409개로 잰 이전 데이터의 값이라 위 값과 비교하지 않는다.
