# 4교시: DS의 일, 2022와 2026

장표: [4교시 DS의 일, 2022와 2026](https://claude.ai/artifact/FvbbskQyEMrxWcMJeRKka8) (HTML: `slides/04_ds_work.html`).

실습 없이 장표로 설명하는 시간이다. 예시 요청 하나 — **“설비 매뉴얼을 자동으로 찾아 주는 AI Agent를 만들어 주세요”**(외부 인터넷 없음, 설비 PC 성능 낮음, 설치는 간단히, 장비 이름 · 명령어를 정확히 몰라도 찾게) — 를 받았다고 하고, 인터뷰와 현황 파악으로 진짜 필요를 가려 **임베딩 검색 모델만 HTML 파일 하나로 배포**하기로 정하는 흐름부터 본다. 그다음 실제 매뉴얼 대신 같은 모양의 공개 데이터인 **법령**으로 과제를 진행한 기록을 따라간다.

다섯 단계마다 손에 남는 산출물의 실제 모양과 DS가 정해야 하는 판단, 같은 일을 2022년과 2026년에 어떻게 하는지를 본다.

| 단계 | 산출물 (실제 기록) | 판단 |
|---|---|---|
| ① 문제 정의 | 인터뷰 질문, 현황 파악, 선택지 비교, 결정 기록 | 정말 Agent가 필요한가 · 목표는 현실적인가 |
| ② 데이터 | 질문 가이드(`.claude/skills/law-question-gen/SKILL.md`), 질문 한 건, 검증 결과 | 데이터가 실제 사용자를 닮았나 |
| ③ 학습 실험 | 실험 계획서(`runs/001_hard_negatives/02_plan.md`), 설정 파일, 장부(`runs/ledger.tsv`) | 무엇을 바꾸고 무엇과 비교할까 |
| ④ 평가 · 분석 | 질문 하나 해부, 틀린 질문 묶기(`runs/001_hard_negatives/00_error_analysis.md`) | 이 차이를 믿을까 |
| ⑤ 공유 · 보고 | 보고서 고치기 전후, 건네는 방법(`runs/001_hard_negatives/04_results.md`, `05_retro.md`) | 무엇을 말하고 뺄까 |

방법의 원리와 숫자 해석은 5교시부터 직접 해 보며 본다.

## 선택: 받은 질문 데이터 보기

장표의 질문 한 건과 검증 스크립트를 직접 보고 싶으면(API 키 없이 몇 초):

```shell
uv run python scripts/data_version.py pull v1      # 3교시 01_doctor.py --fix를 했다면 이미 받았다
uv run python lecture/04_ds_work/01_questions.py
```

받은 질문 12,232개의 유형 비율과 예시, 검증 결과가 나온다. `--run`을 붙이면 일부러 틀린 질문 파일로 검증이 무엇을 잡는지 본다.

| 파일 | 내용 |
|---|---|
| `01_questions.py` | 질문 유형·테마·keyword 어절 수, 질문 → 정답 → hard negative 예시, 검증 |
| `.claude/skills/law-question-gen/SKILL.md` | 질문 생성 지침 (Claude가 읽는 프롬프트) |
