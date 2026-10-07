# 실습 3 (6교시): 학습 실험과 분석

장표: [6교시 실습 3 학습](https://claude.ai/artifact/HBXjApsvu8jxyQxaaH5tP2). 학습 원리 → 실험 비교(가설은 맞았나) → 오답 분석 순서로 본다. 전체 학습(실험 하나에 35~95분)은 강사가 미리 해 둔 모델을 받아 쓰고, 여기서는 몇 step만 맛본다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다. API 키는 쓰지 않는다.
결과는 `experiments/results/lecture/06_train/`에 쓰고, 받은 모델과 비교표는 바꾸지 않는다.

## 0. 받은 파인튜닝 모델 준비

비교에 쓰는 모델 네 개(002 전체 학습 · 004 LoRA · 006 배치 128 · r001_A 다시 캔 오답)와 그 인덱스를 받는다. 3교시에 받은 r001_A 외에 나머지까지 받는 명령이다.

```shell
uv run python scripts/finetuned_drive.py download --all
uv run python lecture/03_setup/01_doctor.py --quick
```

## 1. 학습 원리와 맛보기

MNRL(배치 안의 다른 정답을 오답으로 쓰기), NO_DUPLICATES, 전체 학습 vs LoRA를 숫자로 보고, 받은 모델들의 학습 기록(시간, epoch별 dev 점수)을 읽는다.

```shell
uv run python lecture/06_train/01_train_taste.py
```

선택: 실험 002 설정 그대로 30 step만 직접 학습한다(약 1.5분). 결과 모델은 임시 폴더에 두었다가 지운다.

```shell
uv run python lecture/06_train/01_train_taste.py --run
```

## 2. 실험 비교: 가설은 맞았나

학습 전 e5와 받은 모델 네 개를 같은 test로 잰다(실험 010). 처음에는 약 3분, 다시 실행하면 저장된 결과를 읽어 몇 초다.

```shell
uv run python lecture/06_train/02_compare_runs.py
```

R@5 학습 전 0.513 → 002 0.669 · 004 LoRA 0.641 · 006 0.659 · r001_A 0.723, 차이 검정(paired-test), 학습 없는 선택지와 나란히 놓은 표가 나온다.

선택: test 300개 표본으로 비교와 검정을 직접 해 본다(1분 안쪽). 표본이 작으면 신뢰구간이 넓어진다.

```shell
uv run python lecture/06_train/02_compare_runs.py --run
```

## 3. 오답 분석: 무엇을 고쳤고 무엇이 남았나

2번의 결과(질문별 순위)를 읽어 학습 전 vs r001_A를 질문 단위로 비교한다. 2번을 먼저 실행한다.

```shell
uv run python lecture/06_train/03_error_analysis.py
```

고친 질문 609 · 새로 틀린 질문 79, 여전히 틀리는 질문의 1위 오답이 정답과 어떤 관계인지(다른 법령, 법률↔시행령 등)를 본다. 실패 목록은 `experiments/results/lecture/06_train/failures_r001_A.jsonl`로 남는다. 이 파일을 AI에게 주고 분류를 맡겨 본다.

선택: 일상 상황 질문 300개를 두 모델로 직접 평가해 같은 분석을 한다(1분 안쪽).

```shell
uv run python lecture/06_train/03_error_analysis.py --run
```

## 선택: 실험 기록을 MLflow로 보기

`.env`에 `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`을 넣으면 `ragkit train` · `evaluate` · `compare`가 실행 기록을 남긴다(없으면 아무것도 남기지 않는다). 서버는 터미널 하나를 따로 열어 띄운다(Ctrl+C로 끝).

```shell
uv run --extra mlflow mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --artifacts-destination mlruns/artifacts --port 5050
```

브라우저에서 http://127.0.0.1:5050. macOS는 5000번 포트를 AirPlay가 쓰므로 5050을 쓴다. 8교시 `05_monitoring.py`가 서버 기동부터 한 번에 보여 준다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_train_taste.py` | 학습 데이터 한 줄, in-batch 점수표와 MNRL 손실, NO_DUPLICATES 근거, 받은 모델의 학습 기록 |
| `02_compare_runs.py` | 실험 iteration, 실험 010 비교표, 유형·테마별, 가설 판정, paired-test, 학습 없는 선택지 표 |
| `03_error_analysis.py` | 질문 유형·테마·단어 겹침별 상승, 고침/새로 틀림, 1위 오답 관계, 한계(31위 밖), AI에게 맡길 분류 요청 |

전체 학습을 직접 하려면 `uv run ragkit train experiments/exp_002_finetuned/config.yaml --output-dir models/finetuned/my_002`처럼 출력 폴더를 바꿔서 한다(약 40분, 받은 모델을 덮어쓰지 않게).
