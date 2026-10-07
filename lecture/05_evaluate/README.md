# 5교시: 데이터 · 평가 · 기준 실험

장표: [5교시 데이터 · 평가 · 기준 실험](https://claude.ai/artifact/EzELtdX3t269cS4DHsmX3p). 데이터(코퍼스 · 질문 · 분할) → 평가 기준과 출발점 → 학습 없는 선택지(LLM 쿼리 확장) → 기준 실험 002 한 바퀴 순서로 본다. 교시가 끝나면 기준선(002, test R@5 0.669)이 정해진다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.
기본 모드는 API 키를 쓰지 않는다. 결과는 `experiments/results/lecture/05_evaluate/`에 쓰고, 받은 파일은 바꾸지 않는다.

## 0. 준비 확인

3교시에 받은 데이터·모델·인덱스로 한다. 빠진 것이 있으면 `--fix`가 받아 준다.

```shell
uv run python lecture/03_setup/01_doctor.py --quick
```

## 1. 데이터: 코퍼스와 분할

4교시에 질문을 만들어 봤다. 여기서는 그 질문이 찾아야 할 코퍼스와, 학습 · 평가를 나누는 분할을 본다.

```shell
uv run python lecture/04_data/01_corpus.py
uv run python lecture/04_data/03_split.py
```

법령 54개 · 검색 문서 25,967개, 긴 조문을 항·호로 나눈 결과(512토큰 넘는 문서 18.6% → 1.0%), train 8,294 · dev 1,412 · test 2,526개와 법령이 겹치지 않는지(질문 단위로 나누면 64.6%가 샌다) 본다. 두 스크립트의 `--run`은 [`04_data/README.md`](../04_data/README.md)에 있다.

## 2. 평가셋과 지표, 학습 전 e5의 출발점

작은 예로 R@k · MRR · nDCG를 직접 계산해 보고, 학습 전 e5를 test 2,526개 전체로 잰다(받은 인덱스를 다시 쓰므로 약 1분).

```shell
uv run python lecture/05_evaluate/01_metrics.py
```

R@5 0.513 · R@10 0.601, 질문 유형별 R@5 keyword 0.725 / question 0.517 / situation 0.403이 나온다.

선택: test·dev에서 각 300개를 뽑아 `ragkit evaluate`를 직접 돌린다. dev에는 복수 정답 판정을 붙인다.

```shell
uv run python lecture/05_evaluate/01_metrics.py --run
```

## 3. 학습 없는 선택지: LLM 쿼리 확장

질문마다 LLM을 한 번 불러 법률 용어 검색어를 덧붙이면 얼마나 좋아지고, 얼마나 드는지 본다. 받은 확장 캐시(test 17개)로 비교하므로 키 없이 돈다.

```shell
uv run python lecture/05_evaluate/02_query_expansion.py
```

17개 표본에서 R@5 0.294 → 0.471, 질문당 LLM 호출 1회 · 평균 2.4초. 17개 중 3개는 오히려 나빠진다.

선택: `.env`에 `GEMINI_API_KEY`가 있으면 질문 3개를 새로 확장해 함께 비교한다(무료 등급 하루 20회 중 3회).

```shell
uv run python lecture/05_evaluate/02_query_expansion.py --run
```

## 4. 기준 실험 002 한 바퀴

가설 → 설정 → 파일럿 → 학습 → dev로 epoch 선택 → test 1회. 학습 원리(MNRL, NO_DUPLICATES)와 실험 002의 설정 파일, 받은 모델의 학습 기록(epoch별 dev R@5 0.694 / 0.700 / 0.702, 40분)을 본다.

```shell
uv run python lecture/06_train/01_train_taste.py
```

파일럿: 실험 002 설정 그대로 30 step만 직접 학습한다(약 1.5분, `uv sync --extra train` 필요). 손실이 줄어드는지, step당 시간으로 전체 학습이 몇 분일지 본다. 결과 모델은 임시 폴더에 두었다가 지운다.

```shell
uv run python lecture/06_train/01_train_taste.py --run
```

전체 학습(약 40분)은 강사가 미리 한 모델(`models/finetuned/exp_002`)을 쓴다. test 결과(R@5 0.513 → 0.669)는 장표로 보고, 6교시 `02_compare_runs.py`가 같은 숫자를 다시 잰다.

## 참고

| 파일 | 내용 |
|---|---|
| `../04_data/01_corpus.py` | 코퍼스 규모, 문서 한 건의 필드, 항·호 분할, 글자·토큰 길이 분포 |
| `../04_data/03_split.py` | 분할 표, 법령 교집합, 질문 단위로 나눌 때 새는 비율 |
| `01_metrics.py` | 평가셋(법령 단위 test), 지표 직접 계산, doc · article · multi 판정, 학습 전 e5 test 평가, 유형·테마별, 틀린 질문 예 |
| `02_query_expansion.py` | 확장 캐시 현황, 질문당 비용(호출·토큰·지연), 17개 표본 비교, 좋아진·나빠진 예, 학습 없는 선택지 표 |
| `../06_train/01_train_taste.py` | 학습 데이터 한 줄, in-batch 점수표와 MNRL 손실, NO_DUPLICATES 근거, 설정 파일, 받은 모델의 학습 기록, `--run` 파일럿 |

- 직접 전체 학습을 하려면 `uv run ragkit train experiments/exp_002_finetuned/config.yaml --output-dir models/finetuned/my_002`처럼 출력 폴더를 바꿔서 한다(약 40분, 받은 모델을 덮어쓰지 않게).
- `GEMINI_API_KEY`는 `.env` 파일에 둔다(`.env.example`을 복사). 셸마다 문법이 다른 환경 변수 명령은 쓰지 않는다.
