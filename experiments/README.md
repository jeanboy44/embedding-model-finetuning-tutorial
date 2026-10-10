# 실험 (experiments/)

모든 실험을 번호 하나로 모은다. 번호는 강의에서 설명하는 순서를 따른다.
학습 전 모델 평가(기준선) → 기본 실험 → 오답 분석으로 찾은 실험 → 나머지는 에이전트가 돈 실험(딸깍) → 배포 · 제품화 실험.

| 번호 | 폴더 | 내용 | 교시 | 결과 |
|---|---|---|---|---|
| 000 | `000_base_eval/` | 학습 전 e5-small 평가 (기준선, 학습 없음) | 5 | test R@5 0.513 |
| 001 | `001_baseline/` | 기본 실험: 가진 질문 데이터로 전체 학습 (배치 32, LLM이 고른 오답 1개) | 5 | test R@5 0.669, dev 복수 정답 R@5 0.712 |
| 002 | `002_hard_negatives/` | 오답 분석으로 찾은 실험: 모델이 헷갈리는 오답을 다시 캐서 3개씩 넣는다. 학습 설정(lr 3e-5, 4 epoch)은 003 수렴 확인에서 고른 값 | 6 | **test R@5 0.723**, dev 복수 정답 R@5 0.761 (대조군 0.720) |
| 003 | `003_convergence/` | 수렴 확인: 001과 같고 epoch 6, lr 3e-5 | 6 | dev 복수 정답 R@5 0.721 (best epoch 4) |
| 004 | `004_lora/` | 001 + LoRA (r16, query/key/value), lr 2e-4 | 6 | test R@5 0.641 |
| 005 | `005_batch128/` | 001과 같고 배치 32 → 128 | 6 | test R@5 0.659 |
| 006 | `006_model_soup/` | 002 run A의 seed를 늘려 가중치 평균 (설계만 하고 뺐다) | 6 | - |
| 007 | `007_colloquial/` | 구어 질문 실험. 다른 worktree에서 진행 중이라 아직 이 브랜치에 없다 | 6 | 진행 중 |
| 008 | `008_deploy_bench/` | 배포 최적화 비교 (torch / ONNX / INT8 / 가지치기 + INT8) | 7 | `results/comparison.md` |
| 009 | `009_int8_granularity/` | INT8 양자화 단위: 텐서 단위 vs 채널별 | 7 | test R@5 0.514 vs 0.458 |
| 010 | `010_agent_skill/` | 에이전트 + CLI 스킬: 학습 전 e5 vs 파인튜닝 e5 (002) | 8 | `results/comparison.md` |
| - | `archive/` | 이름을 그대로 둔 옛 실험: `exp_003_llm_query_expansion`, `exp_005_base_model_comparison`, `exp_007_finetune_comparison`, `exp_008_dev_multi_baseline`, `exp_010_lecture_comparison` | | |

- 설정 하나로 끝나는 실험은 `config.yaml`(과 `results/comparison.md`)만 둔다. 실행은 `just train <폴더>` 또는 `uv run ragkit compare experiments/<폴더>/config.yaml`.
- `archive/exp_010_lecture_comparison`은 강의 비교표(학습 전 / 001 / 004 / 005 / 002)다. 6교시 `02_compare_runs.py`와 3교시 `02_try_product.py`가 이 설정과 결과를 읽는다.
- 결과 JSON은 커밋하지 않는다. 비교표(`results/comparison.md`)만 커밋한다.

## 에이전트가 도는 실험의 기록 (002 · 006 · 007)

0.95 개선 계획(docs/PLAN.md "0.95 개선 계획")의 각 단계를 실험 하나로 보고, 실험마다 독립 폴더를 만든다.
모든 실험은 같은 6단계를 반복한다.

| 순서 | 파일 | 내용 | 전문가 검토 |
|---|---|---|---|
| 0 | `00_error_analysis.md` | 직전까지의 최고 모델의 오답 분석 (dev만 사용) | |
| 1 | `01_research.md` | 이번 실험과 관련된 문헌·근거 | 예 |
| 2 | `02_plan.md` | 가설, 변경점, 통과 기준(dev), 실행 명령, 예상 시간 | 예 |
| 3 | `03_run/` | 실험 설정(config.yaml), 로그, train_meta.json 사본 | |
| 4 | `04_results.md` | 결과표와 해석, 계획 대비 차이 | |
| 5 | `05_retro.md` | 회고: 실험 방법, 로깅 방법, 에러 분석을 더 잘하기 위한 개선. 다음 실험에 반영할 것 | 예 |

- `reviews/`: 전문가 agent의 검토 원문. 검토 의견을 어떻게 반영했는지는 각 문서 끝의 "검토 반영" 절에 적는다.
- `artifacts/`: 상위 k개 검색 결과, 임베딩 배열 같은 큰 파일 (git 제외).
- `03_run/*/train_meta.json`, `eval_dev.json`, `data_digest.json`은 실행 당시 그대로 둔 기록이라 옛 경로(`runs/001_hard_negatives`, `models/finetuned/r001_A` 등)가 남아 있다. 아래 대응표로 읽는다.

규칙:
- 모델 선택과 판단은 dev로만 한다. test는 계획의 마지막에 한 번만 쓴다. dev·test를 쓸 때마다 `ledger.tsv`에 적는다.
- 확증 run은 seed 2개가 기본이다. seed SD는 약 0.006이다(002 대조군). 개선 기준은 Δ ≥ 0.02 그리고 Δ > 2 × seed SD다.
- 주요 지표는 dev 복수 정답 R@5(판정 파일 `data/labels/dev.jsonl`)와 법령 평균 R@5다. dev 표준오차가 약 ±0.012이므로 0.02 미만의 차이는 의미 없음으로 본다.
- 제약: 모델은 multilingual-e5-small 하나다. 학습할 때도 다른 모델은 쓰지 않는다. reranker·BM25도 쓰지 않는다. 판정 작업은 Claude가 한다.
- 모델 가중치는 `models/finetuned/<번호>_<이름>`에 두고 커밋하지 않는다. 한 실험에 run이 여럿이면 채택한 run은 실험 이름 그대로, 나머지는 뒤에 run 이름을 붙인다(`002_hard_negatives_B`).

| 실험 | 계획 단계 | 상태 | dev 복수 정답 R@5 |
|---|---|---|---|
| 001 | 기본 학습 | 완료 | 0.712 |
| 003 | 1. 수렴 확인 | 완료 | 0.721 (best epoch 4) |
| 002 | 2. hard negative 재채굴 | 완료: 채굴 negative 3개(A) 채택 | **0.761** (대조군 0.720, A·B·C 모두 개선) |
| 006 | 3. model soup | 설계만 하고 뺐다 | - |
| 007 | 구어 질문 | 진행 중 (다른 worktree) | - |

## 옛 이름 대응표

2026-10-11에 `experiments/exp_0NN_*`와 `runs/0NN_*` 두 계열을 위 번호 하나로 합쳤다.

| 새 폴더 | 옛 폴더 |
|---|---|
| `000_base_eval` | (새로 만듦. 값은 exp_010 · exp_008의 학습 전 행) |
| `001_baseline` | `experiments/exp_002_finetuned` |
| `002_hard_negatives` | `runs/001_hard_negatives` |
| `003_convergence` | `experiments/exp_009_convergence` |
| `004_lora` | `experiments/exp_004_lora` |
| `005_batch128` | `experiments/exp_006_cached_mnrl` |
| `006_model_soup` | `runs/002_model_soup` |
| `007_colloquial` | `runs/003_colloquial` |
| `008_deploy_bench` | `experiments/exp_008_deploy_bench` |
| `009_int8_granularity` | `experiments/exp_012_int8_granularity` |
| `010_agent_skill` | `experiments/exp_011_agent_skill` |
| `archive/exp_0NN_*` | `experiments/exp_0NN_*` (003 · 005 · 007 · 008_dev_multi_baseline · 010) |
| `README.md`, `ledger.tsv` | `runs/README.md`, `runs/ledger.tsv` |

| 새 모델 이름 (`models/finetuned/`) | 옛 이름 |
|---|---|
| `001_baseline` | `exp_002` |
| `002_hard_negatives` (채택한 run A) | `r001_A` |
| `002_hard_negatives-pruned`, `002_hard_negatives-pruned-int8` | `r001_A-pruned`, `r001_A-pruned-int8` |
| `002_hard_negatives_B`, `002_hard_negatives_C` | `r001_B`, `r001_C` |
| `002_hard_negatives_control_s42` · `_s43` · `_s44` | `r001_control_s42` · `_s43` · `_s44` |
| `003_convergence` | `exp_009` |
| `004_lora` | `exp_004` |
| `005_batch128` | `exp_006` |
| `006_model_soup_A_s43` | `r002_A_s43` |
| `007_colloquial_*` | `r003_*` (예: `r003_A_s42` → `007_colloquial_A_s42`) |

인덱스 파일 이름(`data/processed/index/<모델 키>.sqlite`)의 모델 키는 `<폴더 이름>-<가중치 수정 시각>`이다.
폴더 이름을 바꾸면 인덱스 파일 이름도 같이 바꿔야 다시 만들지 않고 쓴다. `mv`는 수정 시각을 바꾸지 않는다.

### 로컬 폴더 이름 바꾸기 (강사 PC, 저장소 루트에서 한 번)

```bash
cd models/finetuned
for pair in exp_002:001_baseline r001_A:002_hard_negatives r001_A-pruned:002_hard_negatives-pruned \
            r001_A-pruned-int8:002_hard_negatives-pruned-int8 r001_B:002_hard_negatives_B r001_C:002_hard_negatives_C \
            r001_control_s42:002_hard_negatives_control_s42 r001_control_s43:002_hard_negatives_control_s43 \
            r001_control_s44:002_hard_negatives_control_s44 exp_009:003_convergence exp_004:004_lora \
            exp_006:005_batch128 r002_A_s43:006_model_soup_A_s43 r003_A_s42:007_colloquial_A_s42; do
  old=${pair%%:*}; new=${pair#*:}; [ -d "$old" ] && mv "$old" "$new"
done
cd ../../data/processed/index
for f in exp_002-* r001_* exp_009-* exp_004-* exp_006-* r002_* r003_*; do
  [ -e "$f" ] || continue
  new=$(echo "$f" | sed -E 's/^exp_002-/001_baseline-/; s/^r001_control_/002_hard_negatives_control_/; s/^r001_A/002_hard_negatives/; s/^r001_B/002_hard_negatives_B/; s/^r001_C/002_hard_negatives_C/; s/^exp_009-/003_convergence-/; s/^exp_004-/004_lora-/; s/^exp_006-/005_batch128-/; s/^r002_A_s43/006_model_soup_A_s43/; s/^r003_/007_colloquial_/')
  mv "$f" "$new"
done
cd ../../..
```

로컬에만 있는(git 무시) 실험 산출물도 새 폴더로 옮긴다.

```bash
for d in experiments/exp_00{3,5,7}_* experiments/exp_008_dev_multi_baseline experiments/exp_010_lecture_comparison; do
  [ -d "$d" ] && rsync -a "$d/" "experiments/archive/$(basename "$d")/" && rm -rf "$d"
done
for pair in exp_008_deploy_bench:008_deploy_bench exp_011_agent_skill:010_agent_skill exp_012_int8_granularity:009_int8_granularity; do
  old=experiments/${pair%%:*}; new=experiments/${pair#*:}; [ -d "$old" ] && rsync -a "$old/" "$new/" && rm -rf "$old"
done
for pair in 001_hard_negatives:002_hard_negatives 002_model_soup:006_model_soup 003_colloquial:007_colloquial; do
  old=runs/${pair%%:*}; new=experiments/${pair#*:}; [ -d "$old" ] && rsync -a "$old/" "$new/" && rm -rf "$old"
done
rmdir runs 2>/dev/null || true
```

`002_hard_negatives`의 `artifacts/exp_009/`(채굴용 top-k 덤프)가 남아 있으면 `artifacts/003_convergence/`로 이름을 바꾼다.

### Drive 다시 올리기 (강사)

Drive의 zip은 옛 폴더 이름으로 묶여 있어 `scripts/finetuned_models.json`에 `pending_reupload`로 표시해 두었다(그동안 `download`는 그 모델을 건너뛴다).
로컬 폴더 이름을 바꾼 뒤 다시 올리면 각 항목이 새 값(파일 이름, file_id, sha256)으로 바뀐다.

```bash
uv run python scripts/finetuned_drive.py upload 002_hard_negatives 001_baseline 004_lora 005_batch128
uv run python scripts/finetuned_drive.py list      # (재업로드 대기) 표시가 없어야 한다
git add scripts/finetuned_models.json && git commit -m "chore: 새 실험 번호로 파인튜닝 모델 다시 올림"
```

Drive 폴더의 옛 zip(`finetuned-r001_A.zip`, `finetuned-exp_002.zip`, `finetuned-exp_004.zip`, `finetuned-exp_006.zip`)은 다 올린 뒤 브라우저에서 지운다.
가지치기 모델(`002_hard_negatives-pruned-int8`)과 그 인덱스를 Drive로 나눠 줄 때는 `upload-files`로 새 이름 그대로 올린다.
