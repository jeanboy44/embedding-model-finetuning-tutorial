# 실습 1 (4교시): 데이터 만들기

장표: [4교시 실습 1 데이터](https://claude.ai/artifact/LU4tYoQei2KW1AmzcqtEAq). 코퍼스 → 질문(Claude 스킬) → 법령 단위 분할 순서로 본다.

## 스크립트

저장소 루트에서 실행한다. 기본은 받은 산출물로 확인, `--run`은 임시 폴더에서 작게 직접 실행한다(받은 파일은 바꾸지 않는다).

| 파일 | 내용 | `--run` |
|---|---|---|
| `01_corpus.py` | 코퍼스 규모(법령·조문·검색 문서·테마), 문서 한 건의 필드, 항·호 분할, 글자·토큰 길이 분포(512토큰) | 전기 테마만 `scripts/prepare_law_data.py`로 임시 폴더에 받아 받은 코퍼스와 비교 (인터넷, 수 초) |
| `02_questions.py` | 받은 질문의 유형 비율, 테마별 수, keyword 어절 수, 질문 → 정답 → hard negative 예시, 검증 스크립트를 파일 하나·전체에 실행 | 일부러 틀린 질문 파일로 검증이 무엇을 잡는지 보기. `--file`로 내가 만든 파일 검증 |
| `03_split.py` | `split_meta.json`의 train·dev·test 표, 법령이 겹치지 않는지, 질문 단위로 나누면 얼마나 새는지 | `ragkit split`을 임시 폴더에 실행해 받은 분할과 같은지 비교 |

```bash
uv run python lecture/04_data/01_corpus.py [--run]
uv run python lecture/04_data/02_questions.py [--run] [--file data/questions_mine/최저임금법__p01.jsonl]
uv run python lecture/04_data/03_split.py [--run]
```

## 받는 법

코퍼스·질문·분할은 데이터 버전 v1(`data/versions/v1.json`) 하나로 함께 받는다. 로그인은 필요 없다.

```bash
uv run python scripts/data_version.py pull v1     # → data/processed/law_docs.json, data/questions/, data/splits/
uv run python scripts/data_version.py status      # 지금 data/가 v1과 같은지
```

질문 묶음만 따로 받을 수도 있다(장표의 3번 명령). 이때는 `data/law-questions/generated/`에 풀리므로 분할도 그 폴더로 만든다.

```bash
uv run python scripts/law_questions_drive.py download   # → data/law-questions/generated/<법령>__pNN.jsonl
uv run ragkit split data/law-questions/generated         # 법령 단위 train/dev/test → data/splits/
```

코퍼스를 처음부터 만들려면 `uv run python scripts/prepare_law_data.py`(legalize-kr, 전체 테마 수 분).

## 질문 생성은 Claude 스킬로 한다

검색 질문, 정답 조문, hard negative는 스크립트가 아니라 Claude Code 스킬 **`law-question-gen`** 으로 만든다. 스킬은 `.claude/skills/law-question-gen/`에 있고, 저장소 루트에서 Claude Code를 열면 자동으로 잡힌다.

```text
최저임금법 파트 1 질문 데이터 만들어줘. data/questions_mine/ 에 써 줘
```

이렇게 요청하면 스킬이 법령 파트를 읽고, 질문 파일 `<법령>__pNN.jsonl`을 쓰고, 검증 스크립트까지 돌린다. 폴더를 말하지 않으면 `data/questions/`에 쓰므로 받은 질문을 덮어쓰지 않게 다른 폴더를 지정한다(파일 이름은 같아야 검증 스크립트가 법령을 읽는다). 질문을 어떻게 쓰는지(일상 말투, 유형 비율, 검색어 형식 few-shot, hard negative 기준)는 [`SKILL.md`](../../.claude/skills/law-question-gen/SKILL.md)를 읽는다.

| 파일 | 내용 |
|---|---|
| `.claude/skills/law-question-gen/SKILL.md` | 질문 생성 지침 (Claude가 읽는 프롬프트) |
| `.claude/skills/law-question-gen/references/query-style.md` | 실제 검색어(KoAIO) 형식 분석 |
| `.claude/skills/law-question-gen/scripts/law_parts.py` | 법령 파트 목록·내용 보기 (`list`, `show 최저임금법 --part 1`) |
| `.claude/skills/law-question-gen/scripts/validate_questions.py` | 질문 파일 검증 |

전체 230개 파트를 다 만들려면 약 2,200만 토큰이 든다. 실습에서는 파트 하나만 직접 만들어 보고, 나머지는 강사가 Drive에 올린 질문을 받아 쓴다.
