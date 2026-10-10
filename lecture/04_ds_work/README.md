# 실습 1 (4교시): 데이터 만들기

장표: [4교시 DS의 일, 2022와 2026](https://claude.ai/artifact/FvbbskQyEMrxWcMJeRKka8). 4교시 실습은 0(데이터 받기)과 2(질문, 특히 "직접 만들어 보기")다.
1(코퍼스)과 3(분할)은 5교시에 데이터를 자세히 볼 때 실행한다([`05_evaluate/README.md`](../05_evaluate/README.md)).

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다. API 키는 쓰지 않는다.
기본은 받은 산출물로 확인하고, `--run`은 임시 폴더에서 작게 직접 실행한다(받은 파일은 바꾸지 않는다).

## 0. 데이터 받기

코퍼스·질문·분할은 데이터 버전 v1 하나로 함께 받는다(`data/versions/v1.json`). 로그인은 필요 없다. 3교시에 `01_doctor.py --fix`를 했다면 이미 받았다.

```shell
uv run python scripts/data_version.py pull v1
uv run python scripts/data_version.py status
```

`status`가 v1과 같다고 하면 된다.

## 1. 코퍼스: 법령 조문을 검색 문서로

```shell
uv run python lecture/04_data/01_corpus.py
```

법령 54개 · 조문 11,723개 · 검색 문서 25,967개, 긴 조문을 항·호로 나눈 결과(512토큰 넘는 문서 18.6% → 1.0%)를 본다.

선택: 전기 테마만 직접 받아 만들어 받은 코퍼스와 같은지 비교한다(인터넷 필요, 수 초).

```shell
uv run python lecture/04_data/01_corpus.py --run
```

## 2. 질문: Claude 스킬이 만든 질문 · 정답 · hard negative

받은 질문 12,232개의 유형 비율, 예시, hard negative를 보고 검증 스크립트를 돌린다.

```shell
uv run python lecture/04_data/02_questions.py
```

검증이 무엇을 잡는지 보려면 일부러 틀린 질문 파일로 돌려 본다. 틀린 줄마다 어떤 오류로 잡히는지 출력된다.

```shell
uv run python lecture/04_data/02_questions.py --run
```

### 직접 만들어 보기 (파트 하나)

1. 파트 목록에서 법령과 파트 번호를 고른다.

   ```shell
   uv run python .claude/skills/law-question-gen/scripts/law_parts.py list
   ```

2. 저장소 루트에서 Claude Code를 열고(`claude`) 이렇게 요청한다. 스킬 `law-question-gen`이 자동으로 잡힌다.

   ```text
   최저임금법 파트 1 질문 데이터 만들어줘. data/questions_mine/ 에 써 줘
   ```

   폴더를 말하지 않으면 `data/questions/`에 써서 받은 질문을 덮어쓴다. 꼭 `data/questions_mine/`을 지정한다.

3. 만든 파일을 검증한다.

   ```shell
   uv run python lecture/04_data/02_questions.py --run --file data/questions_mine/최저임금법__p01.jsonl
   ```

## 3. 분할: 질문이 아니라 법령으로 나눈다

```shell
uv run python lecture/04_data/03_split.py
```

train 8,294 · dev 1,412 · test 2,526개, 세 분할의 법령이 겹치지 않는지, 질문 단위로 나누면 얼마나 새는지(64.6%) 본다.

선택: `ragkit split`을 임시 폴더에 직접 돌려 받은 분할과 같은지 비교한다(몇 초).

```shell
uv run python lecture/04_data/03_split.py --run
```

## 참고

| 파일 | 내용 |
|---|---|
| `01_corpus.py` | 코퍼스 규모, 문서 한 건의 필드, 항·호 분할, 글자·토큰 길이 분포 |
| `02_questions.py` | 질문 유형·테마·keyword 어절 수, 질문 → 정답 → hard negative 예시, 검증 |
| `03_split.py` | `split_meta.json`의 분할 표, 법령 교집합, 질문 단위로 나눌 때 새는 비율 |
| `.claude/skills/law-question-gen/SKILL.md` | 질문 생성 지침 (Claude가 읽는 프롬프트) |
| `.claude/skills/law-question-gen/references/query-style.md` | 실제 검색어(KoAIO) 형식 분석 |

- 질문 묶음만 따로 받을 수도 있다: `uv run python scripts/law_questions_drive.py download` → `data/law-questions/generated/`. 이때 분할은 `uv run ragkit split data/law-questions/generated`.
- 코퍼스를 처음부터 만들려면 `uv run python scripts/prepare_law_data.py`(legalize-kr, 전체 테마 수 분). 받은 코퍼스를 덮어쓰므로 실습 중에는 하지 않는다.
- 전체 230개 파트를 다 만들려면 약 2,200만 토큰이 든다. 실습에서는 파트 하나만 직접 만들어 보고, 나머지는 받은 질문을 쓴다.
