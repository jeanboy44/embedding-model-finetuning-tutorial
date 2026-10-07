# 실습 2 (5교시): 평가 — 학습 없이 어디까지 되나

장표: [5교시 실습 2 평가](https://claude.ai/artifact/ELMMxTeWq6z3uPQcXWVR6U). 지표 → 학습 전 모델의 출발점 → 학습 없이 쓸 수 있는 선택지(LLM 쿼리 확장) 순서로 본다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.
기본 모드는 API 키를 쓰지 않는다. 결과는 `experiments/results/lecture/05_evaluate/`에 쓰고, 받은 파일은 바꾸지 않는다.

## 0. 준비 확인

3교시에 받은 데이터·모델·인덱스로 한다. 빠진 것이 있으면 `--fix`가 받아 준다.

```shell
uv run python lecture/03_setup/01_doctor.py --quick
```

## 1. 평가셋과 지표, 학습 전 e5의 출발점

작은 예로 R@k · MRR · nDCG를 직접 계산해 보고, 학습 전 e5를 test 2,526개 전체로 잰다(받은 인덱스를 다시 쓰므로 약 1분).

```shell
uv run python lecture/05_evaluate/01_metrics.py
```

R@5 0.513 · R@10 0.601, 질문 유형별 R@5 keyword 0.725 / question 0.517 / situation 0.403이 나온다.

선택: test·dev에서 각 300개를 뽑아 `ragkit evaluate`를 직접 돌린다. dev에는 복수 정답 판정을 붙인다.

```shell
uv run python lecture/05_evaluate/01_metrics.py --run
```

## 2. 학습 없는 선택지: LLM 쿼리 확장

질문마다 LLM을 한 번 불러 법률 용어 검색어를 덧붙이면 얼마나 좋아지고, 얼마나 드는지 본다. 받은 확장 캐시(test 17개)로 비교하므로 키 없이 돈다.

```shell
uv run python lecture/05_evaluate/02_query_expansion.py
```

17개 표본에서 R@5 0.294 → 0.471, 질문당 LLM 호출 1회 · 평균 2.4초. 17개 중 3개는 오히려 나빠진다.

선택: `.env`에 `GEMINI_API_KEY`가 있으면 질문 3개를 새로 확장해 함께 비교한다(무료 등급 하루 20회 중 3회).

```shell
uv run python lecture/05_evaluate/02_query_expansion.py --run
```

## 참고

| 파일 | 내용 |
|---|---|
| `01_metrics.py` | 평가셋(법령 단위 test), 지표 직접 계산, doc · article · multi 판정, 학습 전 e5 test 평가, 유형·테마별, 틀린 질문 예 |
| `02_query_expansion.py` | 확장 캐시 현황, 질문당 비용(호출·토큰·지연), 17개 표본 비교, 좋아진·나빠진 예, 학습 없는 선택지 표 |

- 다음 교시(6교시)에 이 표에 파인튜닝 모델 한 줄을 붙인다.
- `GEMINI_API_KEY`는 `.env` 파일에 둔다(`.env.example`을 복사). 셸마다 문법이 다른 환경 변수 명령은 쓰지 않는다.
