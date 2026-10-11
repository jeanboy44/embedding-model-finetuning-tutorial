# 4교시: DS의 일, 2022와 2026

장표: [4교시 DS의 일, 2022와 2026](https://claude.ai/artifact/FvbbskQyEMrxWcMJeRKka8) (HTML: `slides/04_ds_work.html`).

실습 없이 장표로 설명하는 시간이다. 예시 요청 하나 — **“설비 매뉴얼을 자동으로 찾아 주는 AI Agent를 만들어 주세요”**(외부 인터넷 없음, 설비 PC 성능 낮음, 설치는 간단히, 장비 이름 · 명령어를 정확히 몰라도 찾게) — 를 받았다고 하고, 인터뷰와 현황 파악으로 진짜 필요를 가려 **임베딩 검색 모델만 HTML 파일 하나로 배포**하기로 정하는 흐름부터 본다. 그다음 실제 매뉴얼 대신 같은 모양의 공개 데이터인 **법령**으로 과제를 진행한 기록을 따라간다.

다섯 단계마다 손에 남는 산출물의 실제 모양과 DS가 정해야 하는 판단, 같은 일을 2022년과 2026년에 어떻게 하는지를 본다.

| 단계 | 산출물 (실제 기록) | 판단 |
|---|---|---|
| ① 문제 정의 | 인터뷰 질문, 현황 파악, 선택지 비교, 결정 기록 | 정말 Agent가 필요한가 · 목표는 현실적인가 |
| ② 데이터 | 질문 가이드(`.claude/skills/law-question-gen/SKILL.md`), 질문 한 건, 검증 결과 | 데이터가 실제 사용자를 닮았나 |
| ③ 학습 실험 | 실험 계획서(`experiments/002_hard_negatives/02_plan.md`), 설정 파일, 장부(`experiments/ledger.tsv`) | 무엇을 바꾸고 무엇과 비교할까 |
| ④ 평가 · 분석 | 질문 하나 해부, 틀린 질문 묶기(`experiments/002_hard_negatives/00_error_analysis.md`) | 이 차이를 믿을까 |
| ⑤ 공유 · 보고 | 보고서 고치기 전후, 건네는 방법(`experiments/002_hard_negatives/04_results.md`, `05_retro.md`) | 무엇을 말하고 뺄까 |

방법의 원리와 숫자 해석은 5교시부터 직접 해 보며 본다.

## 선택: 받은 질문 데이터 보기

장표 ② 데이터 단계의 질문 한 건과 검증 결과를 내 PC에서 직접 보고 싶으면 아래를 실행한다. 모델을 쓰지 않으므로 1초 안에 끝난다.

```shell
uv run python scripts/data_version.py pull v1      # 3교시 01_doctor.py --fix를 했다면 이미 받았다
uv run python lecture/04_ds_work/01_questions.py
```

스크립트는 `data/processed/law_docs.json`(코퍼스)과 `data/questions/*.jsonl`(질문)을 `ragkit.data.load_corpus` · `load_questions`로 읽고 네 단계를 보여 준다. 둘 중 하나라도 없으면 받는 명령만 알려 주고 끝난다.

1. **규모.** 질문 파일 수와 법령 수, 질문 수, 정답으로 한 번 이상 쓰인 문서 수를 센다. 파일(파트) 230개 · 법령 54개 · 질문 12,232개가 나오고, 정답 문서는 9,704개로 코퍼스 25,967개의 37%다. 마지막 줄의 다섯 필드(`query`, `positive_id`, `hard_negative_ids`, `query_type`, `answer`)가 질문 한 건의 생김새다.
2. **유형과 테마.** `query_type`별 개수와 비율, 정답 문서의 테마별 개수, keyword 질문의 어절 수를 센다. situation 41% · question 36% · keyword 23%이고, 테마는 tax 4,308개가 가장 많고 electric 645개가 가장 적다. keyword 질문 2,815개는 평균 5.6어절이고 2~3어절은 9%뿐이다. 실제 검색어는 2~3어절이 70%라는 장표의 숫자와 견주어, 생성 가이드를 왜 고쳤는지 보는 자리다.
3. **예시: 질문 → 정답 → hard negative.** `최저임금법__p01.jsonl`에서 유형별 질문 하나씩과, 장표에 쓴 "수습 3개월 시급" 질문 한 건을 꺼내 정답 조문(제5조 최저임금액), 오답 두 건(제6조 제1항, 제7조), 기준 답을 나란히 보여 준다. 오답이 정답과 같은 주제인데 질문에는 답하지 않는 조문이라는 것을 눈으로 확인한다. 이어서 질문당 hard negative 수 분포가 나온다. 1개 2,451건 · 2개 9,772건 · 3개 9건이다.
4. **검증.** 스킬의 검증 스크립트 `validate_questions.py`를 서브프로세스로 두 번 돌린다. 먼저 `최저임금법__p01.jsonl` 한 파일(질문 46개, 통과), 다음 230개 파일 전체다. 전체 검사는 가치 낮은 조문 경고(WARN) 1건만 내고 통과한다. 파일 사이에 같은 질문이 다른 정답을 가리키는 중복은 전체를 함께 검사해야 잡히므로 두 번 돌린다.

끝에 질문 데이터를 직접 만들어 보는 Claude Code 명령과 5교시로 이어지는 안내가 나온다. 질문 생성 자체는 이 스크립트가 아니라 Claude Code에서 스킬을 불러 하고, 받은 질문을 덮어쓰지 않도록 `data/questions_mine/` 같은 다른 폴더를 지정한다.

### `--run`: 검증 스크립트가 무엇을 잡는지 직접 보기

```shell
uv run python lecture/04_ds_work/01_questions.py --run
```

기본 네 단계 뒤에 5단계가 붙는다. `최저임금법__p01.jsonl`의 앞 세 줄을 복사해 일부러 틀린 줄 4개(질문에 조문 번호 "제5조" 넣기, 코퍼스에 없는 `positive_id`, 정답을 오답 목록에 넣기, `answer` 필드 빼기)와 정상 줄 1개를 임시 폴더에 쓰고 검증을 돌린다. ERROR 4건이 줄 번호와 함께 찍히고 검증 스크립트가 종료 코드 1(실패)을 낸다. 임시 파일은 바로 지우고 `data/questions/`는 건드리지 않는다. 1초 안에 끝난다.

내가 스킬로 만든 파일을 검증하려면 `--file`을 붙인다. 파일 이름이 `<법령>__pNN.jsonl` 형식이어야 검증 스크립트가 법령을 읽는다.

```shell
uv run python lecture/04_ds_work/01_questions.py --run --file data/questions_mine/최저임금법__p01.jsonl
```

검증 스크립트가 오류로 잡는 것은 필드 누락, 코퍼스에 없는 id, 파일 이름과 다른 법령의 정답, 질문 속 조문 번호(정답 누설), 파일 안과 파일 사이의 중복 질문이다. 너무 짧거나 긴 질문과 가치 낮은 조문은 경고만 낸다.

| 파일 | 내용 |
|---|---|
| `01_questions.py` | 질문 유형·테마·keyword 어절 수, 질문 → 정답 → hard negative 예시, 검증. `--run`이면 일부러 틀린 파일로 검증이 잡는 오류를 보고, `--file`로 내 파일을 검증 |
| `.claude/skills/law-question-gen/SKILL.md` | 질문 생성 지침 (Claude가 읽는 프롬프트) |
