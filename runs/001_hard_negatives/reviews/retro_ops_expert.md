# 001 회고 검토 — 실험 운영·로깅·재현성 (ops 전문가)

검토 대상: `05_retro.md`, `04_results.md`, `03_run/`, `*.py`, `run_one.sh`, `train_cli.py`. 학습은 돌리지 않았다.

## 1. 04_results.md의 숫자를 재현·감사할 수 있는가

결론: **dev 숫자는 eval_dev.json이 있어서 다시 계산할 수 있다. 하지만 "어떤 코드와 어떤 입력으로 그 모델을 만들었는가"는 증명할 수 없다.**

| 빠진 것 | 현재 상태 | 영향 |
|---|---|---|
| run별 실행 명령과 시작·종료 시각 | run_one.sh에만 있다. 실제로 실행한 인자, 환경변수, 재시도 여부는 남지 않았다 | 체인 루프로 돌린 run의 실행 순서와 재시도를 감사할 수 없다 |
| run별 git commit과 dirty 여부 | data_digest.json에 commit 하나(`cefeeb6`)만 있다. 16:50에 한 번 기록했다 | A·B·C를 학습할 때 코드가 바뀌었는지 알 수 없다 |
| 환경 | 라이브러리 3개 버전만 있다. `uv.lock` 해시, Python, macOS, MPS 정보가 없다 | MPS는 비결정적이라 같은 seed로도 수치가 달라질 수 있다. 그 크기를 모른다 |
| 입력 해시와 run의 연결 | train_meta의 `data.digest`는 질문 해시(12자)다. `negatives_file`의 sha256은 run 쪽에 없다 | negatives_B가 판정 수정 뒤에 다시 만들어졌다면 알아챌 수 없다 |
| 출력 모델 해시 | `models/finetuned/r001_*`의 safetensors 해시가 없다. exp_009만 있다 | 다음 실험의 채굴 모델(A)이 이 표의 A와 같은 파일인지 증명할 수 없다 |
| decision.json을 만든 코드 | 스크립트가 없다. 손으로 계산했다 | Δ는 대조군 평균 기준, CI와 p는 s42 기준이라 기준이 섞여 있다. 감사할 수 없다 |
| 간이 평가 대 공식 평가 | train_meta `dev_after` 0.7557, 공식 0.761 | 표에서 어느 쪽인지 각주로만 구분된다 |
| 판정 provenance | 출력 행에는 `qid, cand_id, label, reason`만 있다. 계획에 적은 "모델, 프롬프트 해시"가 없다. 프롬프트 원문도 저장하지 않았다 | B와 judge_summary를 재현할 수 없다 |
| 빠진 81문항 | train_meta는 `train_examples` 8,213개만 남긴다. 어떤 qid가 빠졌는지는 남지 않는다 | 학습 집합이 조용히 바뀌었다 (`_replace_negatives`는 negative가 빈 질문을 막지 않는다) |
| 비용 | 토큰과 시간이 "대략 수백만"으로만 적혀 있다 | 판정을 다시 할지 정할 근거가 없다 |

샤드별 판정 분포를 확인했다 (`artifacts/judge/out`). full 비율은 1.0~15.6%, **partial 비율은 10.2~39.1%**다. b_23~b_29에서 partial이 높다. 2차 wave의 영향이라면 기준이 바뀐 것이다(drift). 이를 확인할 shard→wave→agent 기록이 없다.

## 2. 대규모 병렬 LLM 판정 프로토콜

| 단계 | 규칙 |
|---|---|
| 0. Go/No-go 파일럿 | 300쌍을 판정하고 짧게 학습(≤1 epoch, 또는 그 300쌍에 대한 효과 추정)한다. 판정이 negative 선택을 몇 % 바꾸는지 미리 계산한다. 바뀌는 비율 × 예상 효과가 0.02 미만이면 전체 판정을 하지 않는다. 계획에 사전 등록한다 |
| 1. 격리 | agent마다 `judge/work/<shard>/`를 쓰고, 입력은 읽기 전용으로 둔다. 보조 스크립트는 agent가 쓰지 않는다. 오케스트레이터가 고정된 `judge_io.py`를 제공한다 |
| 2. Manifest | `judge/manifest.json`: 프롬프트 파일과 sha256, 모델 ID, shard별 입력 sha256·행 수·qid 목록 해시, 출력 sha256, agent ID, 시작·종료 시각, 토큰 |
| 3. 행 단위 provenance | 출력 행마다 `shard, agent_id, prompt_sha, model, input_row_sha`를 넣는다. `input_row_sha`는 질문과 후보 텍스트의 해시다. **id가 아니라 내용 해시로 검증해야** 다른 샤드 내용으로 판정한 오염(이번의 96줄)을 잡을 수 있다 |
| 4. Gold·calibration | 40쌍의 gold set(쌍둥이 조문, 다른 법의 같은 규칙 같은 경계 사례 포함)을 만든다. 모든 shard에 섞어 넣는다(id는 숨긴다). shard별 gold 정확도가 85% 미만이면 그 shard를 다시 판정한다 |
| 5. Drift 검사 | shard별 라벨 분포를 χ² 검정하고 wave별로 비교한다. 이상치 shard(전체 비율에서 ±2 SE를 벗어난 것)는 이중 판정 표본을 늘린다 |
| 6. 이중 판정·adjudication | 50쌍이 아니라 5%(≈450쌍)를 shard별로 층화해 이중 판정한다. 불일치는 세 번째 판정(다른 프롬프트 순서)으로 정한다. κ는 shard별로 보고한다 |
| 7. 비용 | manifest에 토큰과 시간을 합산한다. 04_results에 "판정 1쌍당 비용"과 "dev +0.01당 비용"을 적는다 |

## 3. 추가할 재사용 스크립트·템플릿 (최소 세트)

| 파일 | 하는 일 |
|---|---|
| `runs/_template/` | 00~05 md 뼈대, `03_run/run_one.sh`, `reviews/.keep`. 02_plan 뼈대에 "결정 규칙(JSON)", "빈 negative 처리 규칙", "판정 Go/No-go"를 필수 칸으로 넣는다 |
| `runs/_tools/provenance.py` | `provenance.json`을 쓴다: argv, 시작·종료 시각, git commit과 `git status --porcelain` 해시, `uv.lock` sha256, Python·OS·torch·MPS 정보, 입력 파일 sha256(config, negatives_file, splits, labels), 출력 model.safetensors sha256 |
| `run_one.sh` 수정 | 학습 전에 `provenance.py start`, 끝나면 `provenance.py end`를 부르고 `03_run/run_summary.tsv`에 한 줄을 덧붙인다(run, seed, commit, 입력 negatives 해시, 학습 행 수, 빠진 질문 수, best epoch, 간이·공식 R@5, Δ, p, 시간, 모델 해시). 마지막에 `touch $d/DONE` 또는 `FAILED`를 남긴다 |
| `runs/_tools/validate_judgments.py` | 입력·출력 manifest를 대조한다: (qid, cand_id) 커버리지, 중복, `input_row_sha` 일치(내용 오염), 라벨 값, shard별 분포·gold 정확도·χ² drift 표. 실패하면 0이 아닌 코드로 끝난다. 판정 직후 자동으로 돌린다 |
| `runs/_tools/decision.py` | 02_plan의 결정 규칙(`03_run/decision_rule.json`으로 사전 등록)과 eval/paired JSON을 읽어 Holm, Δ, CI, 법령 조건을 계산한다. `decision.json`에 입력 해시를 함께 쓴다. 손 계산을 없앤다 |
| `ragkit train` 수정 (1줄 수준) | `_replace_negatives`에서 negative가 빈 질문 수와 qid를 train_meta에 쓴다. 0보다 크면 `--allow-empty-negatives` 없이 멈춘다 |
| `runs/_tools/queue.sh` | 셸 루프로 파일을 기다리는 대신 run 목록을 순서대로 실행한다. 각 run의 DONE/FAILED, 경과 시간을 `queue.log`에 남긴다. 모니터는 30분짜리 log tail 대신 `run_summary.tsv`와 `queue.log`만 보면 된다 |

## 회고 문서에 대한 의견

- 회고의 "판정 직후 검증 자동화"와 "summary.tsv"는 맞다. 다만 **id 검증만으로는 이번 오염을 잡을 수 없다**는 점을 명시해야 한다. 내용 해시가 필요하다.
- "처음 몇 샤드를 보고 기준을 확정"은 gold set과 drift 검사로 구체화해야 한다. 그렇지 않으면 2차 wave의 partial 상승 같은 drift가 반복된다.
- 판정 비용 교훈은 "Go/No-go 파일럿을 사전 등록"으로 규칙화해야 한다.
- MPS의 비결정성 때문에 seed 3개의 범위(0.0014)가 곧 재현 오차다. 재학습 수치가 이 범위 안에 들면 "재현됨"으로 본다고 README에 적을 것을 권한다.
