set shell := ["bash", "-euc"]

# 로그를 파일로 보낼 때도 loss가 바로 보이게 한다 (버퍼링 끔)
export PYTHONUNBUFFERED := "1"

# 파인튜닝 실험 설정 (docs/PLAN.md "학습 실험 시나리오")
train_experiments := "exp_002_finetuned exp_004_lora exp_006_cached_mnrl"
comparison := "experiments/exp_007_finetune_comparison/config.yaml"

[default]
help:
    @just --list

[group('finetune')]
[doc("질문 폴더로 분할 → 002·004·006 학습 → exp_007 비교표까지 한 번에 실행한다. 질문 데이터만 바꿔 재현할 때 쓴다")]
finetune-suite questions="data/questions": (split questions)
    for exp in {{train_experiments}}; do just train "$exp"; done
    just compare

[group('finetune')]
[doc("질문을 법령 단위로 train/dev/test에 나눈다 (seed 42)")]
split questions="data/questions":
    uv run ragkit split {{questions}}

[group('finetune')]
[doc("실험 하나를 학습한다 (예: just train exp_002_finetuned)")]
train exp *args:
    uv run ragkit train --config experiments/{{exp}}/config.yaml {{args}}

[group('finetune')]
[doc("base e5 / Gemma / 학습한 모델을 test로 비교한다 (결과: exp_007 results/comparison.md)")]
compare:
    uv run ragkit compare --config {{comparison}}
