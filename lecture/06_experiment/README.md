# 실습 2 (6교시): 실험 두 개 — 001은 손으로, 002는 딸깍

장표: [6교시 실험 두 개: 001은 손으로, 002는 딸깍](https://claude.ai/artifact/MtviA3WTikxFfG4a7YrN43). 5교시에 학습한 기본 실험 001(test R@5 0.669)을 직접 판정하고 오답을 쪼개 다음 가설을 찾은 뒤, 그 가설을 에이전트가 한 바퀴 돈 실험 002의 기록을 본다. 우리가 세운 가설 전부(004 LoRA · 005 배치 128 · 003 수렴 · 002 · model soup)는 표로만 보고, model soup은 설계만 하고 뺀 이유를 본다.

수강생은 001만 돌린다. 학습은 하지 않고 5교시에 받은 모델(`models/finetuned/001_baseline`)을 쓴다. 002는 학습 6개와 판정 9,009쌍이 든 실험이라 기록만 읽는다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다. API 키는 쓰지 않는다.
결과는 `experiments/results/lecture/06_experiment/`에 쓰고, 받은 모델과 비교표는 바꾸지 않는다.

## 0. 준비 확인

001 모델은 5교시에, 002 모델(002_hard_negatives)은 3교시에 받았다. 빠진 것이 있으면 `--fix`가 받아 준다.

```shell
uv run python lecture/03_setup/01_doctor.py --quick
```

## 1. 가설 표와 002 판정: 정말 올랐나

우리가 세운 가설과 결과를 한 표로 보고(model soup을 뺀 이유 포함), 학습 전 e5와 001을 test 2,526개로 잰 뒤 질문 단위 대응 검정(`ragkit paired-test`)으로 판정한다. 처음에는 약 3분, 다시 실행하면 저장된 결과를 읽어 몇 초다.

```shell
uv run python lecture/06_experiment/01_compare_runs.py
```

R@5 0.513 → 0.669, Δ +0.156, 95% CI [+0.137, +0.174], 고침 521 · 새로 틀림 128, 나빠진 법령 0/16이 나온다.

선택: test 300개 표본으로 평가와 검정을 직접 해 본다(1분 안쪽). 표본이 작으면 신뢰구간이 넓어진다(CI [+0.103, +0.203]).

```shell
uv run python lecture/06_experiment/01_compare_runs.py --run
```

## 2. 002 오답 분석: 다음 가설 찾기

1번의 결과(질문별 순위)를 읽어 어디서 올랐고 무엇이 여전히 틀리는지, 학습 데이터에 우리가 준 오답(hard negative)이 어땠는지 본다. 1번을 먼저 실행한다.

```shell
uv run python lecture/06_experiment/02_error_analysis.py
```

일상 말투(situation) +0.186, 단어 겹침 하위 25% 0.195 → 0.429. 여전히 틀리는 837개의 1위 오답은 51%가 다른 법령 문서이고, 정답의 60%는 6~30위에 있다. 그런데 우리가 준 오답은 100% 같은 법에서 골랐다. 다음 가설 "모델이 헷갈리는 조문을 오답으로 주면 오른다"를 적는다. 실패 목록은 `experiments/results/lecture/06_experiment/failures_001_baseline.jsonl`로 남는다. 이 파일을 AI에게 주고 분류를 맡겨 본다.

선택: train 질문 300개를 002로 검색해 우리가 준 오답이 몇 위에 오는지 직접 잰다(1분 안쪽). 41%가 50위 밖, 이미 쉬운 오답이다.

```shell
uv run python lecture/06_experiment/02_error_analysis.py --run
```

## 3. 실험 001 기록 보기: 에이전트가 돈 바퀴

2번의 가설을 에이전트가 한 바퀴 돈 기록(`experiments/002_hard_negatives`)을 읽는다. 돌리는 것은 없다(몇 초).

```shell
uv run python lecture/06_experiment/03_hard_negatives.py
```

대조군 seed 3개 0.720(범위 0.0014) → 모델이 캔 오답 3개 0.761(dev 복수 정답 R@5), test 0.723. 계획에 미리 적은 기준, 검토가 잡은 누수, 사람이 정한 기준 · 제약 · 의심 · 규칙을 본다.

## 선택: 실험 기록을 MLflow로 보기

`.env`에 `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`을 넣으면 `ragkit train` · `evaluate` · `compare`가 실행 기록을 남긴다(없으면 아무것도 남기지 않는다). 서버는 터미널 하나를 따로 열어 띄운다(Ctrl+C로 끝).

```shell
uv run --extra mlflow mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --artifacts-destination mlruns/artifacts --port 5050
```

브라우저에서 http://127.0.0.1:5050. macOS는 5000번 포트를 AirPlay가 쓰므로 5050을 쓴다. 8교시 `extra_monitoring.py`가 서버 기동부터 한 번에 보여 준다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_compare_runs.py` | 가설 표(model soup을 뺀 이유 포함), 학습 전 vs 001 test 평가, paired-test 판정. `--run` test 300개 |
| `02_error_analysis.py` | 유형·테마·단어 겹침별 상승, 고침/새로 틀림, 1위 오답 관계, 한계(31위 밖), 우리가 준 오답의 출처와 순위, AI에게 맡길 분류 요청. `--run` train 300개로 오답 순위 |
| `03_hard_negatives.py` | 실험 002 기록: 문서 여섯 단계, 사전 등록 기준, run 6개 결과와 검정, 판정 9,009쌍, 사람이 남는 곳 |

LoRA(실험 004)는 가설 표의 한 줄로만 본다. 설정은 `experiments/004_lora/config.yaml`, 결과는 `experiments/archive/exp_010_lecture_comparison/results/comparison.md`에 있다.
