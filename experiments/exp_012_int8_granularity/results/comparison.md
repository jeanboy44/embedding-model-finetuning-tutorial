# int8_granularity

질문 2526개 (`data/splits/test.jsonl`, 데이터 버전 `a0058b5ae8d9`), 코퍼스 `/Users/jeanboy/workspace/embedding-model-finetuning-tutorial/data/processed/law_docs.json` 전체 대상 검색.

| 모델 | R@5 | multi R@5 | 법령 평균 R@5 | R@1 | R@10 | MRR@10 | nDCG@10 | article R@5 | LLM 호출/질문 | 확장 지연 s | 백엔드 | 차원 | 배포 가능 | 학습 시간 | 학습 파라미터 | best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | **0.513** | 0.513 | 0.540 | 0.284 | 0.601 | 0.381 | 0.434 | 0.577 | 0 | - | onnx | 384 | 예 | - | - | - |
| multilingual-e5-small-int8 | **0.514** | 0.514 | 0.542 | 0.287 | 0.601 | 0.383 | 0.435 | 0.572 | 0 | - | onnx | 384 | 예 | - | - | - |
| multilingual-e5-small-int8-tensor | **0.458** | 0.458 | 0.482 | 0.250 | 0.546 | 0.340 | 0.389 | 0.529 | 0 | - | onnx | 384 | 예 | - | - | - |

## 질문 유형별 R@5 (doc)

| 모델 | keyword | question | situation |
|---|---|---|---|
| intfloat/multilingual-e5-small | 0.725 (n=542) | 0.517 (n=911) | 0.403 (n=1073) |
| multilingual-e5-small-int8 | 0.723 (n=542) | 0.521 (n=911) | 0.403 (n=1073) |
| multilingual-e5-small-int8-tensor | 0.661 (n=542) | 0.468 (n=911) | 0.349 (n=1073) |

## 테마별 R@5 (doc)

| 모델 | consumer | electric | finance | tax | traffic | youth |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 0.496 (n=240) | 0.777 (n=157) | 0.569 (n=494) | 0.456 (n=768) | 0.505 (n=493) | 0.468 (n=374) |
| multilingual-e5-small-int8 | 0.508 (n=240) | 0.771 (n=157) | 0.571 (n=494) | 0.454 (n=768) | 0.505 (n=493) | 0.471 (n=374) |
| multilingual-e5-small-int8-tensor | 0.442 (n=240) | 0.726 (n=157) | 0.520 (n=494) | 0.396 (n=768) | 0.473 (n=493) | 0.385 (n=374) |
