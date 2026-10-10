# 도구와 규칙

명령은 모두 저장소 루트에서 실행한다.

## ragkit

| 하는 일 | 명령 |
|---|---|
| 분할 (이미 있음) | `uv run ragkit split data/questions --unit article --test-ratio 0.15 --dev-ratio 0.1 --out data/splits_article` |
| 학습 | `uv run ragkit train <config.yaml> --splits data/splits_article [--output-dir ...] [--max-steps 30 --limit 300 --no-dev-eval]` |
| 평가 | `uv run ragkit evaluate <모델 이름|폴더> --split dev --splits data/splits_article --out <json> [--labels <판정 파일>]` |
| 대응 검정 | `uv run ragkit paired-test <기준 json> <비교 json> --judge doc --k 5 --out <json>` |
| 여러 모델 비교표 | `uv run ragkit compare <비교 config.yaml>` |

- 학습 설정의 출발점은 `experiments/001_baseline/config.yaml`이다. 복사해서 바꾸는 것만 고친다.
- 파일럿: 본 학습 전에 `--max-steps 30 --limit 300 --no-dev-eval --output-dir <임시>`로 메모리 · step당 시간 · 손실 방향을 본다.
- 평가는 코퍼스 임베딩을 `data/processed/index/`에 저장해 재사용한다. 학습한 모델은 가중치 수정 시각이 키에 들어가므로 다시 학습하면 새 인덱스를 만든다.

## 재사용할 스크립트 (002에서 만든 것)

`experiments/archive/002_hard_negatives/`에 있다. 새 실험 폴더로 복사해 경로 상수(`ART`, 모델 이름, 분할 경로)만 고쳐 쓴다. 002는 법령 단위 분할 시절이라 `data/splits` · `data/labels/dev.jsonl`을 가리키는 곳을 확인한다.

| 파일 | 하는 일 |
|---|---|
| `dump_topk.py` | 한 분할의 질문마다 상위 50개 결과와 코퍼스 · 질문 임베딩 저장 (오답 분석 · 채굴 재료) |
| `error_analysis.py` | 실패를 1위 오답 관계 6유형으로 집계, 정답 순위 분포, 기존 hard negative 순위 |
| `mine.py` | 학습 모델로 train 질문마다 train 문서 중 상위 후보 채굴, 정답 점수비로 가짜 negative 후보 표시 |
| `compare_runs.py` | 두 run의 실패 유형 비교, 고친 · 망친 질문 목록 |
| `common.py` | 위 스크립트가 같이 쓰는 로더와 관계 분류 |
| `03_run/run_one.sh` | run 하나 학습 → 평가 → 검정을 이어 붙인 예 (새 실험에서는 결과를 `runs/<실험>/<run>/`에 쓰게 고친다) |

## 폴더 · 이름

- 실험 폴더: `experiments/<번호>_<이름>/` (번호는 README 표의 다음 빈 번호, 이름은 짧은 영어 snake_case)
- 모델: `models/finetuned/<번호>_<이름>`, run이 여럿이면 채택 run은 실험 이름 그대로, 나머지는 `_<run>`(예: `002_hard_negatives_B`, `002_hard_negatives_control_s42`). 커밋하지 않는다.
- 실행 결과 전부: `runs/<번호>_<이름>/` (SKILL.md "기록" 절). 문서는 `experiments/<번호>_<이름>/`
- MLflow 실험 이름: `loop/<번호>_<이름>`. 올리기: `uv run --extra mlflow python .claude/skills/experiment-loop/scripts/log_experiment.py <번호>_<이름>`
- 결과 JSON은 커밋하지 않고, 표와 문서만 커밋한다. 커밋은 사용자가 요청할 때만 한다.

## ledger.tsv

`experiments/ledger.tsv`, 탭 구분, 열: `date experiment split comparison metric value decision`.

```
2026-10-12	013 run 선택	dev	A · B · C vs 대조군	doc R@5	A 0.___ / 대조군 0.___	A 채택 (seed 2개 확인)
2026-11-01	반복 종료 · 최종 모델	test	학습 전 / 001 / 최종	doc R@5	0.___ / 0.___ / 0.___	반복 종료 후 단 한 번. 이후 이 숫자로 아무것도 고르지 않음
```

dev를 쓸 때(run 고르기)마다 적는다. test는 반복이 모두 끝난 뒤 한 번만 쓰고, 그 한 줄을 적는다. decision 열에는 그 숫자로 무엇을 정했는지 쓴다.

## README 갱신

바퀴를 마치면 `experiments/README.md`의 실험 표(번호 · 폴더 · 내용 · 교시 · 결과)와 진행 표(실험 · 계획 단계 · 상태 · dev R@5)에 한 줄씩 더한다.
