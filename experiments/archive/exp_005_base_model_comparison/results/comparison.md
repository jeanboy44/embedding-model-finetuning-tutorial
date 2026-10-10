# base_model_comparison

질문 409개, 코퍼스 `/Users/jeanboy/workspace/embedding-model-finetuning-tutorial/.claude/worktrees/ragkit/data/processed/law_docs.json` 전체 대상 검색.

| 모델 | 백엔드 | 차원 | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | article R@10 | 배포 가능 |
|---|---|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | onnx | 384 | 0.296 | 0.526 | 0.611 | 0.396 | 0.448 | 0.655 | 예 |
| google/embeddinggemma-300m | st | 768 | 0.582 | 0.817 | 0.873 | 0.678 | 0.725 | 0.890 | 예 |

## 테마별 R@10 (doc)

| 모델 | electric | youth |
|---|---|---|
| intfloat/multilingual-e5-small | 0.682 (n=44) | 0.603 (n=365) |
| google/embeddinggemma-300m | 0.955 (n=44) | 0.863 (n=365) |
