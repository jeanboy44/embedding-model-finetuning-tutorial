# 법령 질문 데이터

임베딩 모델(MNRL) 학습·평가용 질문–정답 문서 쌍이다. `law-question-gen` 스킬로 만들었다.
이 파일은 Drive 묶음(`law-questions.zip`)의 README 원본이다(`scripts/law_questions_drive.py bundle`이 zip에 넣는다).

## 받기

```bash
uv run python scripts/law_questions_drive.py download      # → data/law-questions/ (로그인 불필요)
```

## 폴더 구조

```
law-questions/
├── README.md
├── generated/                 # 학습·평가용 데이터: <법령>__pNN.jsonl, 파트 하나가 파일 하나
├── questions/                 # 스킬 개발 중 테스트 결과 보관용 (학습에 쓰지 않음)
│   ├── questions.jsonl
│   └── raw/
└── corpus/
    ├── law_docs.json                # 현재 코퍼스: 긴 조문을 항·호 단위로 나눔 (문서 25,967개)
    └── law_docs_article_level.json  # 나누기 전 코퍼스: 조 단위 (questions/의 iteration-1용)
```

## generated/: 학습·평가에는 이 폴더를 쓴다

- 질문 12,232개(정답 문서 9,704개), 파일 230개(2026-09-30 기준). 54개 법령의 모든 파트다.
  - 테마별: tax 4,308 / finance 2,322 / traffic 2,273 / youth 1,791 / consumer 893 / electric 645
  - 유형: situation 41% / question 36% / keyword 23%. 질문에 법령 이름이 들어간 비율은 1%다.
- 행정 조문이 많은 파트(조세특례제한법, 자본시장법 등)는 질문이 적고, 생활 밀착형 법령 파트는 많다.
- 파일을 모두 함께 검증해 `validate_questions.py` 오류 0을 확인했다. 파일 사이에 중복 질문도 없다.
- id는 모두 `corpus/law_docs.json` 기준이다.

| 필드 | 설명 |
|---|---|
| `query` | 검색 질문 (일상 말투) |
| `positive_id` | 정답 문서 id |
| `hard_negative_ids` | 같은 파트 안에서 주제는 비슷하지만 답하지 못하는 문서 1~3개 |
| `query_type` | `situation` / `question` / `keyword` |
| `answer` | 정답 문서에 근거한 한두 문장 답 (RAG 평가용) |

```python
import json, glob
rows = [json.loads(l) for f in sorted(glob.glob("data/law-questions/generated/*.jsonl")) for l in open(f)]
```

- 임베딩할 문서 텍스트는 `title + "\n" + text`로 한다.
- e5 접두어(`query: `, `passage: `)는 데이터에 들어 있지 않으니 코드에서 붙인다.
- 같은 정답 문서에 질문이 여러 개 붙어 있다. MNRL 학습 시 `BatchSamplers.NO_DUPLICATES`를 쓴다.
- 학습·평가 분할은 법령 또는 테마 단위로 한다. 질문 단위로 나누면 같은 조문의 질문이 양쪽에 들어간다.

## questions/: 스킬 개발 테스트 결과 (참고용)

스킬을 만들면서 비교 실험한 결과 1,138개다. 레코드마다 `source`(`{iteration, eval, config}`)와 `corpus`(id를 찾을 코퍼스 파일)가 붙어 있다.
이 중 품질이 되는 것(iteration-2, `with_skill`: 최저임금법, 도로교통법 파트 1, 전기공사공제조합법)은 이미 `generated/`에 들어 있다. 나머지는 아래 이유로 학습에 쓰지 않는다.

- **`iteration: 1`:** 나누기 전 조 단위 id라서 현재 코퍼스와 맞지 않는다.
- **`config: without_skill`:** 스킬 없이 만든 비교용 데이터다. `query_type`·`answer`가 없는 것이 있고, 행정 조문 질문이 많다.
