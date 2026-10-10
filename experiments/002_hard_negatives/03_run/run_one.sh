#!/usr/bin/env bash
# 사용: run_one.sh <run 이름>  — 학습 → train_meta 복사 → 공식 dev 평가 → 대조군(seed 42) 대비 대응 검정
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=runs/001_hard_negatives/03_run; run=$1; d=$R/$run
out=$(grep -E '^\s*output_dir:' $d/config.yaml | awk '{print $2}')
PYTHONUNBUFFERED=1 uv run ragkit train --config $d/config.yaml > $d/train.log 2>&1
cp $out/train_meta.json $d/
PYTHONUNBUFFERED=1 uv run ragkit evaluate $out --split dev --labels data/labels/dev.jsonl --out $d/eval_dev.json > $d/eval.log 2>&1
uv run ragkit paired-test $R/control_s42/eval_dev.json $d/eval_dev.json --out $d/paired_test.json > $d/paired_test.txt 2>&1
