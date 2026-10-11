# 실습 2 (6교시): 실험 두 개 — 001은 손으로, 002는 딸깍

장표: [6교시 실험 두 개: 001은 손으로, 002는 딸깍](https://claude.ai/artifact/MtviA3WTikxFfG4a7YrN43). 5교시에 학습한 기본 실험 001(조 단위 test R@5 0.744, 학습 전 0.545)을 직접 판정하고 오답을 쪼개 다음 가설을 찾은 뒤, 그 가설을 에이전트가 한 바퀴 돈 실험 002의 기록을 본다. 001 위에 세운 가설 전부(002 오답 다시 캐기 · 003 오답 3개 · 004 · 005 맥락 헤더 · 006 model soup · 007 구어 질문)는 표로만 보고, model soup은 설계만 하고 뺀 이유를 본다.

수강생은 001만 돌린다. 학습은 하지 않고 5교시에 받은 모델(`models/finetuned/001_baseline`)을 쓴다. 002는 학습 run 4개(파일럿 1개 + 본 run 3개, run 하나 38~70분)와 train 9,126개 채굴이 든 실험이라 기록만 읽는다.

평가 분할은 5교시와 같은 조 단위 분할(`data/splits_article/`, test 1,863개)이다. 54개 법령이 모두 학습에 있고, 학습 질문이 없던 조만 test에 있다. 법령 단위 분할(`data/splits/`)은 쓰지 않는다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다. API 키는 쓰지 않는다.
결과는 `experiments/results/lecture/06_experiment/`에 쓰고, 받은 모델과 결과 JSON은 바꾸지 않는다.

## 0. 준비 확인

001 모델은 5교시에, 002 모델(`002_hard_negatives`)은 3교시에 받았다. 조 단위 분할은 5교시 `02_split.py`가 만들었다. 빠진 것이 있으면 `--fix`가 받아 준다.

```shell
uv run python lecture/03_setup/01_doctor.py --quick
```

## 1. 가설 표와 001 판정: 정말 올랐나

우리가 세운 가설과 결과를 한 표로 보고(model soup을 뺀 이유 포함), 학습 전 e5와 001을 test 1,863개로 잰 결과를 질문 단위 대응 검정(`ragkit paired-test`)으로 판정한다. 강사 결과 JSON(`experiments/results/001_baseline/`)은 Drive로 나눠 주지 않으므로, 처음에는 두 모델을 직접 평가해 약 2분이고 다시 실행하면 그 결과를 읽어 몇 초다.

```shell
uv run python lecture/06_experiment/01_compare_runs.py
```

스크립트가 하는 일:

1. 가설 표를 출력한다. 001은 오늘 직접 판정하고, 002~005는 에이전트가 dev로만 판정한 기록이다(002 맞음 · 채택, 003 · 004 · 005 기각, 006 설계만, 007 진행 중). 판정 규칙은 "0.02 미만 차이는 차이로 보지 않는다"(dev 표준오차 약 ±0.012)이고, 002부터는 test를 쓰지 않았다는 것을 본다.
2. 학습 전 e5와 001의 조 단위 test 결과를 읽는다. 받은 결과(`experiments/results/001_baseline/`)가 있으면 그것을 읽고, 없으면 `ragkit evaluate … --splits data/splits_article`로 두 모델을 평가해 `experiments/results/lecture/06_experiment/001_baseline/`에 쓴다. 출력의 R@1 · R@5 · R@10 · MRR@10 표에서 R@5 0.545 → 0.744를 본다.
3. `ragkit paired-test … --judge doc`으로 같은 질문끼리 짝지어 검정한다. Δ · 95% CI · 고침/망침 · 나빠진 법령 수를 보고, CI가 0을 걸치지 않으면 차이로 본다.

R@5 0.545 → 0.744, Δ +0.199, 95% CI [+0.177, +0.222], 고침 448 · 망침 77, 나빠진 법령 0/54가 나온다(실험 001).

선택: test 1,863개에서 300개를 뽑아(seed 42) 임시 폴더에 쓰고, 같은 `ragkit evaluate` · `paired-test`를 직접 돌린다(받은 인덱스를 다시 쓰므로 1분 안팎). 표본이 작으면 같은 차이라도 신뢰구간이 넓어지는 것을 본다. 결과는 임시 폴더에만 쓴다.

```shell
uv run python lecture/06_experiment/01_compare_runs.py --run
```

## 2. 001 오답 분석: 다음 가설 찾기

1번이 읽은 결과(질문별 순위 · top-10)로 어디서 올랐고 무엇이 여전히 틀리는지, 학습 데이터에 우리가 준 오답(hard negative)이 어땠는지 본다. 1번을 먼저 실행한다(몇 초).

```shell
uv run python lecture/06_experiment/02_error_analysis.py
```

스크립트가 하는 일:

0. 1번과 같은 순서로 결과 JSON을 찾아 읽는다(받은 결과 → 1번이 직접 평가한 결과). 질문 1,863개 · 법령 54개 · 판정 doc(정답 문서 하나)을 확인한다.
1. 결과 JSON의 유형 · 테마 · 법령별 집계를 학습 전 → 001로 비교하고, 질문과 정답 조문의 단어 겹침(`ragkit.evaluation.query_overlap`, 글자 2-gram)으로 4분위를 나눠 직접 센다. 일상 말투(situation) 0.417 → 0.671(+0.254), 겹침 하위 25% 0.191 → 0.517을 본다. 파인튜닝 뒤에도 겹침이 적을수록 낮다.
2. `ragkit.evaluation.compare_hits`로 같은 질문끼리 짝지어 고침 448 · 새로 틀림 77 · 둘 다 틀림 400 · 둘 다 맞힘 938으로 나누고 예시를 본다. 새로 틀린 질문의 60%는 6~10위로 살짝 밀린 것이다.
3. 001이 여전히 틀리는 477개의 1위 오답과 정답의 관계를 `confusion_relation`으로 매긴다. 다른 법령 문서가 1위인 경우 42%, 같은 법의 법률↔시행령 혼동 30%이고, 정답의 63%는 6~30위에 있다. 틀린 situation 질문 5개와 법률↔시행령 혼동 예를 읽는다.
4. 정답이 31위 밖인 실패 178개(37%)는 오답을 바꾸는 학습으로 닿지 않는다는 것과, 그 밖의 한계(말투 차이 · 추론 · 코퍼스에 없는 지식)를 본다. test 조는 학습에 없었지만 54개 법령은 모두 학습에 있다.
5. train 9,126개에 붙은 오답 14,360개가 100% 정답과 같은 법에서 골라졌음을 센다. 기본 모드는 실험 002 오답 분석의 기록(001로 train 표본 1,000개 검색, 기존 오답의 39%가 이미 50위 밖)을 보여 준다.
6. 실패 477개를 `experiments/results/lecture/06_experiment/failures_001_baseline.jsonl`로 남기고, AI에게 맡길 분류 요청과 다음 가설 "모델이 헷갈리는 조문을 오답으로 주면 오른다"를 적는다.

이 파일을 AI에게 주고 분류를 맡겨 본다. DS는 어떤 숫자를 믿을지와 다음에 무엇을 할지를 정한다.

선택: train 질문 300개(오답이 붙은 것만, seed 42)를 001 모델과 받은 인덱스로 직접 검색해 우리가 준 오답이 몇 위에 오는지 잰다(1분 안쪽). 50위 밖 오답은 학습해도 배울 것이 거의 없는 쉬운 오답이다.

```shell
uv run python lecture/06_experiment/02_error_analysis.py --run
```

## 3. 실험 002 기록 보기: 에이전트가 돈 바퀴

2번의 가설을 에이전트가 한 바퀴 돈 기록(`experiments/002_hard_negatives`)을 읽는다. 돌리는 것은 없다(몇 초).

```shell
uv run python lecture/06_experiment/03_hard_negatives.py
```

스크립트가 하는 일:

1. 바퀴의 여섯 단계가 문서 하나씩(`00_error_analysis.md` ~ `05_retro.md`)으로 남은 것과 검토 문서 2개(검색 · 방법론 전문가 역할 에이전트)를 나열한다. 학습 로그 `03_run/`은 커밋하지 않아 이 저장소에 없다.
2. `02_plan.md`에 결과를 보기 전에 적은 가설 · 근거 · 고정 · 기준 · 의심 · 탐색을 본다. 기준은 Δ ≥ 0.02 · seed 범위 초과 · CI 하한 > 0 · 나빠진 법령 0 · dev-b 동반 상승 · seed 2개 확인이고, 검토가 짚은 누수(채굴 후보의 19.6%가 dev · test 정답 조 문서)를 후보 풀에서 뺐다.
3. `04_results.md`의 결과표를 읽어 공용 대조군 seed 3개 0.7144(범위 0.0040) → A1 0.7534(Δ +0.0390, CI [+0.0209, +0.0531]), 탐색 run A3 0.7611을 출력하고, 판정 기준 7개를 A1이 모두 통과해 채택한 것을 본다. 배포 형태(가지치기 + INT8, 29.7MB) dev R@5 0.7546. test는 아직 쓰지 않았다(dev로만 골랐다).
4. 사람이 정한 기준 · 제약 · 의심 · 규칙(`05_retro.md`: 다음 바퀴의 대조군은 A1, 무거운 설정은 파일럿부터)을 2번에서 수강생이 직접 한 일과 견준다.

## 선택: 실험 기록을 MLflow로 보기

`.env`에 `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`을 넣으면 `ragkit train` · `evaluate` · `compare`가 실행 기록을 남긴다(없으면 아무것도 남기지 않는다). 서버는 터미널 하나를 따로 열어 띄운다(Ctrl+C로 끝).

```shell
uv run --extra mlflow mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --artifacts-destination mlruns/artifacts --port 5050
```

브라우저에서 http://127.0.0.1:5050. 8교시 `extra_monitoring.py`가 서버 기동부터 한 번에 보여 준다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_compare_runs.py` | 가설 표(model soup을 뺀 이유 포함), 학습 전 vs 001 조 단위 test 평가, paired-test 판정. `--run` test 300개 |
| `02_error_analysis.py` | 유형·테마·단어 겹침별 상승, 고침/새로 틀림, 1위 오답 관계, 한계(31위 밖), 우리가 준 오답의 출처와 순위, AI에게 맡길 분류 요청. `--run` train 300개로 오답 순위 |
| `03_hard_negatives.py` | 실험 002 기록: 문서 여섯 단계, 사전 등록 기준, 공용 대조군 3 seed와 A1 · A3 결과표, 판정 기준 7개, 사람이 남는 곳 |
