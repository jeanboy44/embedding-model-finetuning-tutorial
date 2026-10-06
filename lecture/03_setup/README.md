# 실습 0 (3교시): 준비와 완성품 먼저 써 보기

장표: [3교시 실습 0 준비](https://claude.ai/artifact/VuQyTKs7BHKu4Hsq5xy5Qc). 환경 점검 → 웹(강사 시연) → CLI로 질문 넷 → 학습 전후 비교 순서로 본다.
API 키는 쓰지 않는다. 명령은 모두 저장소 루트에서 실행하고, Mac 터미널과 Windows PowerShell에서 그대로 돈다.

## 1. 환경 점검

```bash
uv sync --all-packages --all-extras
uv run python lecture/03_setup/01_doctor.py --fix   # 빠진 데이터 · 모델 · 인덱스를 받고 다시 점검
```

`--fix`가 대신 돌리는 받기 명령은 아래와 같다. 로그인은 필요 없다.

```bash
uv run python scripts/data_version.py pull v1       # 코퍼스 · 질문 · 분할
uv run python scripts/download_model_hf.py          # 학습 전 e5 (HF가 막히면 download_model_gdrive.py)
uv run python scripts/finetuned_drive.py download   # 파인튜닝 r001_A + 인덱스 두 개 (학습 전 e5 · r001_A)
```

## 2. 질문 넷을 학습 전 모델에 던져 본다

정답으로 떠야 할 조문이 위쪽에 뜨는지 본다.

| 질문 | 말투 | 찾아야 할 조문 |
|---|---|---|
| 야간 근로 수당 | 키워드 | 근로기준법 제56조 |
| 편의점 알바 3개월 했는데 주휴수당 받을 수 있나요? | 상황 설명 | 근로기준법 제55조 |
| 회사 그만뒀는데 퇴직금은 언제까지 받아야 해요? | 궁금한 점 | 근로자퇴직급여 보장법 제9조 |
| 방문판매로 산 정수기 환불하고 싶어요 | 상황 설명 | 방문판매법 제8조 |

```bash
uv run --package ragkit-search ragkit-search search "야간 근로 수당"
uv run --package ragkit-search ragkit-search search "편의점 알바 3개월 했는데 주휴수당 받을 수 있나요?"
uv run --package ragkit-search ragkit-search search "회사 그만뒀는데 퇴직금은 언제까지 받아야 해요?"
uv run --package ragkit-search ragkit-search search "방문판매로 산 정수기 환불하고 싶어요"

uv run --package ragkit-search ragkit-search show 근로기준법_법률_제55조   # 조문 원문
```

## 3. 학습 전후 비교: 모델만 바꾼다

같은 명령 뒤에 `--checkpoint models/finetuned/r001_A --backend torch`만 붙이면 파인튜닝한 모델(r001_A)로 찾는다. 인덱스는 자동으로 그 모델 것을 쓴다.

```bash
uv run --package ragkit-search ragkit-search search "편의점 알바 3개월 했는데 주휴수당 받을 수 있나요?" --checkpoint models/finetuned/r001_A --backend torch
uv run --package ragkit-search ragkit-search search "회사 그만뒀는데 퇴직금은 언제까지 받아야 해요?" --checkpoint models/finetuned/r001_A --backend torch
uv run --package ragkit-search ragkit-search search "방문판매로 산 정수기 환불하고 싶어요" --checkpoint models/finetuned/r001_A --backend torch
```

말투만 바꿔서도 비교해 본다. 학습 전 모델은 법률 용어로 물어야 찾고, 파인튜닝한 모델은 평소 말투로도 찾는다.

```bash
uv run --package ragkit-search ragkit-search search "주휴일 유급휴일 1주 개근"                        # 학습 전 · 법률 용어
uv run --package ragkit-search ragkit-search search "편의점 알바를 3개월 했는데 주휴수당을 받을 수 있나요?"        # 학습 전 · 일상어
uv run --package ragkit-search ragkit-search search "편의점 알바를 3개월 했는데 주휴수당을 받을 수 있나요?" --checkpoint models/finetuned/r001_A --backend torch   # 파인튜닝 · 일상어
```

강사 PC에서 잰 정답 순위 (조 단위, 전체 2.6만 문서 중):

| 질문 | 학습 전 | 파인튜닝 |
|---|---|---|
| 야간 근로 수당 | 2위 | 3위 |
| 편의점 알바 3개월 했는데 주휴수당 받을 수 있나요? | 65위 | 1위 |
| 회사 그만뒀는데 퇴직금은 언제까지 받아야 해요? | 5위 | 2위 |
| 방문판매로 산 정수기 환불하고 싶어요 | 100위 밖 | 18위 |
| 주휴일 유급휴일 1주 개근 (법률 용어) | 3위 | — |

근로기준법은 학습에 쓴 법령이라 체감용 예시다. 처음 보는 법령에서의 성적(test R@5 0.513 → 0.723)은 6교시에 본다.

## 스크립트

| 파일 | 내용 |
|---|---|
| `01_doctor.py` | 설치 · 데이터 · 모델 · 인덱스 점검, 받은 모델을 열어 검색해 보는 동작 확인, `--fix`로 받기 (`--quick`이면 동작 확인 생략) |
| `02_try_product.py` | 위 질문 넷을 학습 전 · 파인튜닝 모델에 한 번에 던져 top-3와 정답 순위를 나란히 보여 준다. 웹 · CLI · MCP 띄우는 명령 안내 |
| `extra_gemini_basics.py` | (선택) Gemini API 기초. `GEMINI_API_KEY`가 있어야 하고 한 번에 4회 호출한다 |

웹(법령 노트)은 3교시에 강사 화면으로만 보고, 직접 띄우는 것은 8교시(`lecture/08_product/04_web_app.py`)에서 한다. MCP도 8교시에 다룬다.
