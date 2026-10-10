# 4교시: DS의 일, 2022와 2026

장표: [4교시 DS의 일, 2022와 2026](https://claude.ai/artifact/FvbbskQyEMrxWcMJeRKka8). 임베딩 모델 하나가 나오기까지의 다섯 단계를 펼쳐 보고, 같은 일을 2022년에는 어떻게 했고 2026년에는 무엇을 에이전트에게 맡겼는지 비교한다. 실습은 하나다: **질문 데이터 만들기, 손으로 vs 에이전트로.**

코퍼스와 분할은 5교시에 데이터를 자세히 볼 때 실행한다([`05_data_eval/README.md`](../05_data_eval/README.md)).

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다. API 키는 쓰지 않는다.
기본은 받은 산출물로 확인하고, `--run`은 임시 폴더에서 작게 직접 실행한다(받은 파일은 바꾸지 않는다).

## 0. 데이터 받기

코퍼스·질문·분할은 데이터 버전 v1 하나로 함께 받는다(`data/versions/v1.json`). 로그인은 필요 없다. 3교시에 `01_doctor.py --fix`를 했다면 이미 받았다.

```shell
uv run python scripts/data_version.py pull v1
uv run python scripts/data_version.py status
```

`status`가 v1과 같다고 하면 된다.

## 1. 받은 질문 보기

Claude 스킬이 만든 질문 12,232개의 유형 비율, 예시, hard negative를 보고 검증 스크립트를 돌린다.

```shell
uv run python lecture/04_ds_work/01_questions.py
```

검증이 무엇을 잡는지 보려면 일부러 틀린 질문 파일로 돌려 본다. 틀린 줄마다 어떤 오류로 잡히는지 출력된다.

```shell
uv run python lecture/04_ds_work/01_questions.py --run
```

## 2. 2022 방식: 손으로 만들기 (10분)

1. 최저임금법 파트 1의 조문을 읽는다.

   ```shell
   uv run python .claude/skills/law-question-gen/scripts/law_parts.py show 최저임금법 --part 1
   ```

2. 질문 · 정답 조문 · 헷갈리는 오답을 직접 3개 써서 `data/questions_hand/최저임금법__p01.jsonl`로 저장한다. 한 줄에 질문 하나, 모양은 1번에서 본 받은 질문과 같다.

   ```json
   {"query": "부모님 식당 일을 돕는데 저도 최저시급 받을 수 있나요?", "positive_id": "최저임금법_법률_제3조", "hard_negative_ids": ["최저임금법_법률_제7조"], "query_type": "situation", "answer": "동거하는 친족만 사용하는 사업에는 최저임금이 적용되지 않습니다."}
   ```

3. 검증 스크립트로 확인한다.

   ```shell
   uv run python lecture/04_ds_work/01_questions.py --run --file data/questions_hand/최저임금법__p01.jsonl
   ```

## 3. 2026 방식: 에이전트에게 맡기기 (같은 파트)

1. 저장소 루트에서 Claude Code를 열고(`claude`) 이렇게 요청한다. 스킬 `law-question-gen`이 자동으로 잡힌다.

   ```text
   최저임금법 파트 1 질문 데이터 만들어줘. data/questions_mine/ 에 써 줘
   ```

   폴더를 말하지 않으면 `data/questions/`에 써서 받은 질문을 덮어쓴다. 꼭 `data/questions_mine/`을 지정한다.

2. 같은 검증 스크립트로 확인한다.

   ```shell
   uv run python lecture/04_ds_work/01_questions.py --run --file data/questions_mine/최저임금법__p01.jsonl
   ```

다른 파트를 고르고 싶으면 `uv run python .claude/skills/law-question-gen/scripts/law_parts.py list`로 법령과 파트 번호를 본다.

## 4. 비교해 볼 것

걸린 시간 · 만든 개수 · 검증 오류 · 말투가 정말 일상어인가 · 오답이 정말 헷갈리는가. 뒤의 두 가지는 스크립트가 못 잡는다. 사람이 몇 개를 직접 읽어 본다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_questions.py` | 질문 유형·테마·keyword 어절 수, 질문 → 정답 → hard negative 예시, 검증 |
| `.claude/skills/law-question-gen/SKILL.md` | 질문 생성 지침 (Claude가 읽는 프롬프트) |
| `.claude/skills/law-question-gen/references/query-style.md` | 실제 검색어(KoAIO) 형식 분석 |

- 질문 묶음만 따로 받을 수도 있다: `uv run python scripts/law_questions_drive.py download` → `data/law-questions/generated/`. 이때 분할은 `uv run ragkit split data/law-questions/generated`.
- 전체 230개 파트를 다 만들려면 약 2,200만 토큰이 든다. 실습에서는 파트 하나만 직접 만들어 보고, 나머지는 받은 질문을 쓴다.
