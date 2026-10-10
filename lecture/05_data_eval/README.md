# 실습 1 (5교시): 데이터 · 평가 · 기본 실험

장표: [5교시 데이터 · 평가 · 기본 실험](https://claude.ai/artifact/EzELtdX3t269cS4DHsmX3p). 0부 데이터 만들기(코퍼스 · 분할) → 1부 base 모델 평가(지표 → 학습 전 모델로 기준선) → 2부 1차 기준 모델 학습(기본 실험 001 한 바퀴) 순서로 본다. 교시가 끝나면 기준선(학습 전 e5, test R@5 0.545)과 기본 실험 001(0.744)이 한 표에 놓인다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.
기본 모드는 API 키를 쓰지 않는다. 결과는 `experiments/results/lecture/05_data_eval/`에 쓰고, 받은 파일은 바꾸지 않는다.

## 0. 준비 확인

3교시 `--fix`로 받은 데이터·모델·인덱스로 한다(2부에 쓰는 기본 실험 001 모델 포함). 5교시 줄이 준비됨이 아니면 `--fix`를 붙여 다시 돌린다. 조 단위 분할(`data/splits_article/`)은 받지 않고 질문에서 바로 만든다(몇 초, seed 42라 누가 만들어도 같다).

```shell
uv run python lecture/03_setup/01_doctor.py --quick
```

## 0부 · 데이터 만들기: 코퍼스와 분할

4교시에 질문을 만들어 봤다. 여기서는 그 질문이 찾아야 할 코퍼스와, 학습 · 평가를 나누는 분할을 본다.

```shell
uv run python lecture/05_data_eval/01_corpus.py
uv run python lecture/05_data_eval/02_split.py
```

법령 54개 · 검색 문서 25,967개, 긴 조문을 항·호로 나눈 결과(512토큰 넘는 문서 18.6% → 1.0%), 조 단위 분할 train 9,126 · dev 1,243 · test 1,863개(조 4,389 · 610 · 942개, 54개 법령 모두 세 분할에)와 조가 겹치지 않는지(질문 단위로 나누면 67.1%가 샌다), train 오답에서 dev · test 정답 조의 문서 2,072개를 뺀 것을 본다. 분할이 없으면 이 스크립트가 만든다.

선택: 전기 테마만 직접 받아 코퍼스를 만들어 받은 것과 비교하고(인터넷 필요, 수 초), `ragkit split --unit article`을 임시 폴더에 직접 돌려 같은 분할이 나오는지 비교한다(몇 초).

```shell
uv run python lecture/05_data_eval/01_corpus.py --run
uv run python lecture/05_data_eval/02_split.py --run
```

코퍼스를 처음부터 만들려면 `uv run python scripts/prepare_law_data.py`(legalize-kr, 전체 테마 수 분). 받은 코퍼스를 덮어쓰므로 실습 중에는 하지 않는다.

## 1부 · base 모델 평가: 지표와 학습 전 e5 = 기준선

작은 예로 R@k · MRR · nDCG를 직접 계산해 보고, 학습 전 e5를 test 1,863개 전체로 잰다(받은 인덱스를 다시 쓰므로 약 1분).

```shell
uv run python lecture/05_data_eval/03_metrics.py
```

R@5 0.545 · R@10 0.630, 질문 유형별 R@5 keyword 0.739 / question 0.573 / situation 0.417이 나온다.

선택: test에서 300개를 뽑아 `ragkit evaluate`를 직접 돌린다. dev는 복수 정답 판정이 있는 81개로 돌린다(판정 파일은 옛 dev로 만들어 지금 dev와 일부만 겹친다).

```shell
uv run python lecture/05_data_eval/03_metrics.py --run
```

## 2부 · 1차 기준 모델 학습: 기본 실험 001 한 바퀴

가설 → 설정 → 파일럿 → 학습 → dev로 epoch 선택 → test 1회. 학습 원리(MNRL, NO_DUPLICATES)와 실험 001의 설정 파일, 받은 모델의 학습 기록(dev R@5 학습 전 0.540 → epoch별 0.697 / 0.710 / 0.717, 39분)을 본다.

```shell
uv run python lecture/05_data_eval/04_baseline_train.py
```

파일럿: 실험 001 설정 그대로 30 step만 직접 학습한다(약 1.5분, `uv sync --extra train` 필요). 손실이 줄어드는지, step당 시간으로 전체 학습이 몇 분일지 본다. 결과 모델은 임시 폴더에 두었다가 지운다.

```shell
uv run python lecture/05_data_eval/04_baseline_train.py --run
```

전체 학습(M4 Pro 39분)은 강사가 미리 한 모델(`models/finetuned/001_baseline`)을 쓴다. test 결과(R@5 0.545 → 0.744)는 장표로 보고, 6교시에 판정한다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_corpus.py` | 코퍼스 규모, 문서 한 건의 필드, 항·호 분할, 글자·토큰 길이 분포 |
| `02_split.py` | 조 단위 분할 표(없으면 만든다), 조 교집합, 오답에서 뺀 문서, 질문 단위로 나눌 때 새는 비율 |
| `03_metrics.py` | 평가셋(조 단위 test), 지표 직접 계산, doc · article · multi 판정, 학습 전 e5 test 평가, 유형·테마별, 틀린 질문 예 |
| `04_baseline_train.py` | 학습 데이터 한 줄, in-batch 점수표와 MNRL 손실, NO_DUPLICATES 근거, 실험 001 설정 파일, 받은 모델의 학습 기록, `--run` 파일럿 |

- 직접 전체 학습을 하려면 `uv run ragkit train experiments/001_baseline/config.yaml --splits data/splits_article --output-dir models/finetuned/my_001`처럼 출력 폴더를 바꿔서 한다(약 40분, 받은 모델을 덮어쓰지 않게).
