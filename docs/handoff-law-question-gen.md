# 인수인계: 법령 코퍼스 + 질문 생성 스킬 (2026-09-28)

다음 세션은 이 파일부터 읽고 "남은 작업"을 이어서 한다.
강의 전체 흐름(4단계)은 `docs/PLAN.md`에 있다. 이 작업은 그중 1단계(DS 본업: 데이터·학습·평가)에 속한다.
프로젝트 루트: `/Users/jeanboy/workspace/embedding-model-finetuning-tutorial`
(작업 도중 `~/workspace/slm-finetuning-example`에서 이름이 바뀜.)

## 목표
임베딩 파인튜닝 강의(8시간)용 학습/평가 데이터를 만든다.
- 검색 대상(코퍼스): 대한민국 법령 조문. 출처는 GitHub의 legalize-kr/legalize-kr (법령을 Markdown으로 관리하고, 개정 1건 = 실제 공포일자를 가진 커밋 1개)
- 질문: Claude가 스킬을 사용해 법령 단위로 생성한다. 학생·사회초년생 말투 질문, 정답 조문(positive), hard negative를 만든다.

## 사용자와 합의한 결정 (다시 논의하지 않음)
1. VS Code 문서는 쓰지 않는다. 용어가 너무 일반적이라 베이스 모델이 이미 잘 알고, 파인튜닝 효과가 드러나지 않는다.
2. legalize-kr를 쓰고, 도메인을 특정한다. 전체 법령(3,050개, 약 5천만 토큰)은 쓰지 않는다.
3. 코퍼스는 "LLM 컨텍스트에 한 번에 넣기 어려운" 규모여야 한다. 현재 약 480만 토큰(o200k)으로, 100만 컨텍스트의 약 5배다.
4. 테마는 6개다. electric, youth(사회 첫걸음), traffic, tax, finance, consumer. 고용보험법도 포함한다.
5. 질문은 LLM 한 번 호출로 만들지 않고 **법령·파트 단위**로 만든다. 출력 한도, 품질, 재시도 비용 때문이다.
6. 질문 생성 방식: Claude가 쓸 SKILL.md를 **/skill-creator 절차로** 만들고, 그 스킬로 질문을 만든다(사용자 요청).
7. 학습 손실은 MultipleNegativesRankingLoss(MNRL)다. 정답이 같은 질문이 한 배치에 들어가지 않게 `BatchSamplers.NO_DUPLICATES`를 쓴다.
8. e5 접두어(`query: `, `passage: `)는 학습·평가 소스코드에서 붙인다. 학습 데이터(JSONL)와 코퍼스에는 넣지 않는다.
9. 512토큰을 넘는 긴 조문은 항 단위로, 항도 길면 호 묶음으로 나눈다(2026-09-28). `MAX_CHARS = 800`(제목+본문 글자 수) 기준이고, 삭제된 항·호·목 줄("② 삭제")도 본문에서 지운다. 코퍼스는 조문 11,723개 → 문서 25,967개, 파트 230개가 됐다. 제16항부터 쓰이는 `**<16>** <16>` 표기도 항으로 인식한다(2026-09-29 수정 전에는 중복 id 15개가 있었다). `build_corpus`는 id가 겹치면 멈춘다. 512토큰 초과는 279개(1.1%)만 남았는데, 주로 세법 시행령의 표와 아주 긴 단일 호다. 조각 id는 `…_제6조_제4항`, `…_제2조_제1~19호`이고 `parent_id`가 원래 조문을 가리킨다. 임베딩할 텍스트는 `title + "\n" + text`로 한다(조각은 title에 조 제목과 항 레이블이 들어 있다).

## 완료된 것

### 1. 데이터 준비 스크립트: `scripts/prepare_law_data.py`
- legalize-kr에서 테마별 법령 폴더만 받아(sparse checkout) 조문 단위 JSON을 만든다.
- 실행: `uv run python scripts/prepare_law_data.py` (옵션: `--themes youth`, `--as-of 2020-01-01`)
- 출력: `data/processed/law_docs.json`. 54개 법령, **11,723조**, 24MB(본문 16.7MB, 약 695만 자)
- 원본 저장소: `data/raw/legalize-kr/` (76MB. 그중 .git 51MB는 `--as-of` 테스트로 받은 전체 이력)
- 레코드 필드: `id`(예: `근로기준법_법률_제55조`), `title`(`근로기준법 제55조 (휴일)`), `text`, `category`(법령 폴더 이름), `theme`, `law_name`, `law_type`, `article_no`, `article_title`, `chapter`, `promulgation_date`, `effective_date`, `source_url`
- 파싱 규칙:
  - 같은 제목의 파일이 여러 개면 공포일자가 최신인 파일만 쓴다. `kr/근로기준법/법률.md`는 1997년 폐지 법률이고 현행은 `법률(법률).md`다.
  - 부칙, "삭제 <…>" 조문, "이를 폐지한다" 조문은 뺀다.
  - `<개정 …>`, `<신설 …>`, `**`, `<img>` 태그는 지운다. `[ … ]` 괄호 안은 실제 내용이라 남긴다.
- 테스트: `tests/test_prepare_law_data.py` (3개 통과). `.gitignore`에 `data/raw/*`, `!data/raw/.gitkeep`을 추가했다.
- 주의: legalize-kr는 force-push로 커밋 해시가 바뀔 수 있다. 재현할 때는 해시 대신 `--as-of` 날짜를 기록한다.

테마별 규모 (o200k 토큰):
| 테마 | 조문 | 토큰 |
|---|---|---|
| electric | 773 | 22만 |
| youth | 1,570 | 43만 |
| traffic | 2,566 | 86만 |
| tax | 3,171 | 191만 (조세특례제한법 하나가 117만 자) |
| finance | 2,635 | 110만 |
| consumer | 1,008 | 30만 |

### 2. 질문 생성 스킬: `.claude/skills/law-question-gen/` (진행 중)
- `scripts/law_parts.py` (완료, 동작 확인)
  - `list [--theme X]`: 법령별 조문 수, 파트 수, 완료 여부를 보여 준다. 현재 **파트 218개**, 0개 완료.
  - `show <법령> --part N`: 파트의 조문을 출력한다. 가치가 낮은 조문에는 `[LOW-VALUE]` 표시가 붙는다.
  - 파트는 조문을 순서대로 약 40,000자씩 묶은 것이다(`PART_CHARS`). 출력 경로 규칙은 `data/questions/<법령>__pNN.jsonl`이다.
- `scripts/validate_questions.py` (작성 완료, **실제 JSONL로는 아직 테스트 안 함**)
  - 검사: 필수 필드, positive/negative id가 코퍼스에 있는지, positive가 파일의 법령에 속하는지, negative에 positive가 없는지, query_type 값, 질문에 "제N조"가 들어 있는지(누설), 중복 질문
  - 경고: 질문 길이(5~150자 밖), LOW-VALUE 조문 사용. 통계로 유형 분포와 법령 이름 포함 비율을 보여 준다.
- `SKILL.md`: **아직 안 씀**. 다음 세션 첫 작업이다. 아래 설계를 따른다.

## 남은 작업

### A. SKILL.md 작성: `.claude/skills/law-question-gen/SKILL.md`
설계:
- name: `law-question-gen`
- description(적극적으로 발동되게): 법령 코퍼스(`data/processed/law_docs.json`)로 임베딩 학습/평가용 질문, 쿼리-정답 쌍, hard negative를 만들 때 사용한다. "질문 생성", "쿼리 만들어", "학습 데이터 만들어", 법령 이름과 "질문" 등이 나오면 발동한다.
- 작업 흐름:
  1. `law_parts.py list`로 진행 상황을 확인한다.
  2. `law_parts.py show <법령> --part N`으로 파트를 읽는다.
  3. `data/questions/<법령>__pNN.jsonl`을 쓴다.
  4. `validate_questions.py`로 검사하고, 오류가 0이 될 때까지 고친다.
- 출력 레코드(JSONL 한 줄): `{"query", "positive_id", "hard_negative_ids": [1~3개], "query_type": "situation|question|keyword", "answer": "조문 근거의 한두 문장 답"}`. `answer`는 이후 RAG 평가(exp_003)에 쓴다.
- 질문 원칙(이유를 함께 설명할 것):
  - **어휘 차이가 핵심이다.** 일상 표현으로 묻는다. 예: "편의점 알바 3개월 했는데 주휴수당 받을 수 있어요?" 법률 용어를 그대로 옮기면 베이스 모델도 쉽게 맞혀서 파인튜닝 효과가 사라진다. keyword 유형만 검색어처럼 용어를 써도 된다.
  - 질문자는 테마에 맞춘다. youth는 학생·알바생·사회초년생·자취생, traffic은 운전자·킥보드 이용자, tax·finance·consumer는 직장인·자영업자·소비자, electric은 전기 기술자·사업자로 한다.
  - 조문 번호("제N조")는 쓰지 않는다(누설). 법령 이름은 일부(약 20% 이하)만 쓴다.
  - 그 조문 하나만으로 답할 수 있는 질문을 만든다. 조문에 없는 사실은 지어내지 않는다.
  - 질문 수는 중요도에 따라 조문당 0~3개로 한다. 권리·의무·금액·기간 조문은 2~3개, 절차·행정 조문은 0~1개, 벌칙·과태료는 상황형 1개(예: "신호위반하면 벌금 얼마예요?"), `[LOW-VALUE]` 조문은 보통 0개다.
  - 유형 비율은 대략 situation 50%, question 30%, keyword 20%로 한다.
  - hard negative는 **같은 파트 안에서** 주제가 비슷하지만 그 질문에 답하지 못하는 조문 1~3개로 한다. 같은 조의 시행령·시행규칙처럼 정답이 될 수도 있는 조문은 negative로 쓰지 않는다.
- 규모 안내: 파트 218개를 병렬 서브에이전트로 나눠 처리할 수 있게, "파트 하나 = 작업 하나"로 설명한다.

### B. skill-creator 절차로 테스트 (사용자 지시: /skill-creator 사용)
- 워크스페이스: `.claude/skills/law-question-gen-workspace/iteration-1/`
- 테스트 프롬프트 3개(제안):
  1. "최저임금법 질문 데이터 만들어줘" (youth, 작은 법령, 파트 1개)
  2. "도로교통법 파트 1 질문 생성해줘" (traffic, 큰 법령의 일부)
  3. "전기공사공제조합법으로 학습용 쿼리 만들어" (electric, 행정 조문이 많아 건너뛰기 판단 확인)
- with_skill과 without_skill을 동시에 실행한다. 테스트 출력은 `data/questions/`가 아니라 워크스페이스 `outputs/`에 저장한다.
- assertion(스크립트로 채점): validate 오류 0, 조문 번호 누설 0, hard negative가 전부 실제 id, situation 비율 40% 이상, LOW-VALUE 조문 질문 비율 낮음, answer 필드 존재
- `eval-viewer/generate_review.py`로 사용자 리뷰를 받는다. skill-creator 경로: `/Users/jeanboy/.claude/plugins/cache/claude-plugins-official/skill-creator/fbe07fb6ce7d/skills/skill-creator`

### B-2. 스킬 테스트 진행 상황 (2026-09-28)
- iteration-1: baseline 3개가 이 문서를 읽어 스킬 설계에 오염됨. 결과는 조 단위 id라 현재 코퍼스와 맞지 않는다.
- iteration-2 (항 분할 코퍼스, baseline은 `.claude/`·`docs/` 읽기 금지): with_skill 10/10, baseline 7/10 (baseline 실패는 형식 차이). 스킬은 행정 조문을 건너뛰어 baseline보다 질문이 적고(46 vs 87, 169 vs 189, 44 vs 133) 시행령·항 조각을 정확히 정답으로 쓴다. 채점: `law-question-gen-workspace/grade.py`.
- 사용자 검토 대기 중. 뷰어: `generate_review.py … --previous-workspace iteration-1`.

### B-3. 테스트 질문 모음과 Google Drive (2026-09-28, 완료)
- `scripts/law_questions_drive.py`: `collect`(테스트 질문 → `data/questions_test/`), `bundle`(→ `dist/law-questions.zip` + sha256), `upload`(rclone, remote `gdrive`), `download`(gdown, 로그인 불필요 → `data/law-questions/`)
- Drive 폴더: https://drive.google.com/drive/folders/1y30fzc21thrATVzwBa9vzBWbOSTJt-U1 (업로드·다운로드 왕복, sha256 확인 완료)
- 질문 1,138개. 권장: iteration-2 with_skill 259개. 자세한 내용은 `docs/law-questions-bundle.md`(zip README 원본)
- rclone의 공용 client_id는 2026년 중 지원이 끝난다. 계속 쓰려면 자체 client_id를 만든다(https://rclone.org/drive/#making-your-own-client-id).

### B-4. youth 테마 본 생성 (2026-09-29, 완료)
- 24개 파트 모두 `data/questions/`에 생성했다. 질문 1,791개, 정답 문서 1,289개. 유형 비율은 situation 47%, question 33%, keyword 20%다. 모든 파일을 함께 검증해 오류 0을 확인했다. 최저임금법 파트는 iteration-2 결과를 그대로 썼다.
- 비용은 파트당 평균 약 9.7만 토큰, 전체 약 230만 토큰이다. 파트 하나는 1~7분 걸린다. 서브에이전트 12개를 병렬로 두 번 돌렸다.
- 전체 231개 파트로 추정하면 약 2,200만 토큰, 질문 약 1.7만 개다(테마마다 편차가 있다).
- 파일 사이 중복 질문 검사를 `validate_questions.py`에 추가했다. "청년 나이 기준"이 청년기본법과 청년고용촉진특별법에서 서로 다른 정답을 가리키는 것을 발견해 고쳤다.
- Drive에 업로드했다(zip 안 `generated/`). 테스트 결과 중 iteration-2 with_skill 도로교통법 p01과 전기공사공제조합법 p01도 `data/questions/`로 옮겼다. 그래서 26/231 파트가 완료됐고 질문은 2,004개다.
- `data/questions/`와 `data/questions_test/`는 모두 gitignore 대상이고 Drive에만 보관한다.
- `.claude/` 전체를 git 이력에서 뺐다(2026-09-29). 스킬과 평가 워크스페이스는 `law_questions_drive.py bundle-skill` → `upload --zip-path dist/law-question-gen-skill.zip`으로 Drive에 따로 보관한다(`law-question-gen-skill.zip`). 나중에 `lecture/01_data`로 옮길 예정이다.
- 스킬을 git에 넣었다(2026-10-01). 원본은 `.claude/skills/law-question-gen/`이고, `.gitignore`는 `.claude/` 아래에서 이 폴더만 관리 대상으로 둔다. `lecture/04_data/README.md`(옛 `01_data`)는 스킬을 쓰라고 안내만 한다. Drive zip(`bundle-skill`)에는 이제 평가 워크스페이스만 담는다.

### B-5. 전체 파트 생성 (2026-09-30, 완료)
- 남은 204개 파트를 생성해 230/230 파트가 완료됐다. 새 질문은 10,228개이고, 전체는 질문 12,232개, 정답 문서 9,704개다. 파일을 모두 함께 검증해 오류 0을 확인했다.
- 파트당 목표는 평균 약 50개였다. 행정 조문 위주인 파트(조세특례제한법, 자본시장법)는 20개 안팎이고, 소비자·교통 파트는 60~80개다.
- 유형 비율은 situation 41%, question 36%, keyword 23%다. 세금·금융 쪽 기업 대상 조문 때문에 situation이 목표(50%)보다 낮다.
- 서브에이전트 23개(파트 9개씩)로 돌리다 세션 사용량 한도에 걸려 19개가 멈췄다. 남은 26개 파트는 묶음 7개로 다시 돌렸다.
- 여신전문금융업법 p02는 뒷부분이 비어 있어(쓰다 끊긴 것으로 판단) 질문 14개를 덧붙였다.
- 표본을 뽑아 질문과 답이 조문 내용에 맞는지 직접 읽는 검수는 아직 하지 않았다. 먼저 볼 곳은 조세특례제한법 p13~p21의 창업자금 특례 질문이다. 법률 본문이 파트에 없어 답을 넓게 썼다.
- Drive에 다시 업로드했다(`law-questions.zip`, 9.7MB).

### C. 스킬 확정 후
- ~~231개 파트 전체를 생성한다.~~ 완료(B-5).
- 학습/평가 분할 스크립트를 만든다. 제안은 법령 또는 테마 단위 분할(처음 보는 분야로 일반화하는지 확인)이다.
- 생성한 질문(`data/questions/`)은 git에 넣지 않고(gitignore) Google Drive에 보관한다(사용자 결정, 2026-09-29). 생성한 뒤에는 `law_questions_drive.py bundle` → `upload`를 실행한다.

## 커밋 상태
코드와 문서는 main에 커밋돼 있다. `data/questions/`와 `.claude/`는 git에 넣지 않고 Drive에 보관한다.
