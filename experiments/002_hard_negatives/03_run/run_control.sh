#!/usr/bin/env bash
# 대조군 3개 seed를 순서대로 학습하고, 각 best 모델을 공식 평가한다.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=runs/001_hard_negatives/03_run
for s in 42 43 44; do
  d=$R/control_s$s
  PYTHONUNBUFFERED=1 uv run ragkit train --config $d/config.yaml > $d/train.log 2>&1
  cp models/finetuned/r001_control_s$s/train_meta.json $d/
  PYTHONUNBUFFERED=1 uv run ragkit evaluate models/finetuned/r001_control_s$s --split dev \
    --labels data/labels/dev.jsonl --out $d/eval_dev.json > $d/eval.log 2>&1
done
