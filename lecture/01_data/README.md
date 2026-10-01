# 실습 1 (4교시): 데이터 만들기

## 질문 생성은 Claude 스킬로 한다

검색 질문, 정답 조문, hard negative는 스크립트가 아니라 Claude Code 스킬 **`law-question-gen`** 으로 만든다. 스킬은 `.claude/skills/law-question-gen/`에 있고, 저장소 루트에서 Claude Code를 열면 자동으로 잡힌다.

```text
최저임금법 질문 데이터 만들어줘
도로교통법 파트 1 질문 생성해줘
```

이렇게 요청하면 스킬이 법령 파트를 읽고, `data/questions/<법령>__pNN.jsonl`을 쓰고, 검증 스크립트까지 돌린다. 질문을 어떻게 쓰는지(일상 말투, 유형 비율, 검색어 형식 few-shot, hard negative 기준)는 [`SKILL.md`](../../.claude/skills/law-question-gen/SKILL.md)를 읽는다.

| 파일 | 내용 |
|---|---|
| `.claude/skills/law-question-gen/SKILL.md` | 질문 생성 지침 (Claude가 읽는 프롬프트) |
| `.claude/skills/law-question-gen/references/query-style.md` | 실제 검색어(KoAIO) 형식 분석 |
| `.claude/skills/law-question-gen/scripts/law_parts.py` | 법령 파트 목록·내용 보기 |
| `.claude/skills/law-question-gen/scripts/validate_questions.py` | 질문 파일 검증 |

전체 230개 파트를 다 만들려면 약 2,200만 토큰이 든다. 실습에서는 파트 하나만 직접 만들어 보고, 나머지는 강사가 Drive에 올린 질문을 받아 쓴다.

```bash
uv run python scripts/law_questions_drive.py download
```
