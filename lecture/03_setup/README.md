# 실습 0 (3교시): 준비와 완성품 먼저 써 보기

장표: [3교시 실습 0 준비](https://claude.ai/artifact/VuQyTKs7BHKu4Hsq5xy5Qc). uv 설치와 최종 목표(HTML 파일 하나) → 환경 점검 → 완성품 써 보기 → CLI로 질문 넷 → 학습 전후 비교 순서로 본다.
API 키는 쓰지 않는다. 명령은 모두 저장소 루트에서 실행하고, Mac 터미널과 Windows PowerShell에서 그대로 돈다.

## 0. uv 설치와 최종 목표

### uv 설치

실습에 필요한 파이썬 환경과 패키지는 `uv`로 관리합니다. `uv`가 아직 설치되어 있지 않다면 먼저 설치합니다.

- **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

설치 확인 (새 터미널 또는 새 PowerShell 창):
```bash
uv --version
```

> **Windows 참고**: 파이썬이 한글 데이터를 UTF-8로 다루도록 `setx PYTHONUTF8 1`을 한 번 실행하고 PowerShell 창을 새로 엽니다.  
> 자세한 OS별 안내는 [macOS 설치 가이드](../../INSTALL_MAC.md)와 [Windows 설치 가이드](../../INSTALL_WINDOWS.md)를 참고하세요.

### 설치 전에: 최종 목표

`law-search-offline.html`(56MB)을 더블클릭하면 Python · 서버 · 인터넷 · API 키 없이 브라우저에서 법령 조문 2.6만 건을 검색한다.
오늘 만드는 파인튜닝 모델(정답이 10위 안에 드는 비율 60% → 80%)을 7교시 방법(ONNX · INT8 · 어휘 가지치기)으로 30MB까지 줄여
조문 데이터와 함께 파일 하나에 넣은 것이다. 만드는 법은 [`apps/offline-search/README.md`](../../apps/offline-search/README.md).

가장 먼저, 저장소를 받기도 전에 이 파일부터 받는다. 아래 링크에서 내려받아 더블클릭한다.

**받기: [law-search-offline.html (Google Drive)](https://drive.google.com/file/d/1TzgE1mh2ZSyAjmPFJtb7OTuk05JFr3mt/view?usp=sharing)**

## 1. 환경 점검

```bash
uv sync --all-packages --all-extras
uv run python lecture/03_setup/01_doctor.py --fix   # 3~8교시에 쓸 것을 모두 받고 다시 점검 (처음 한 번, 약 1.5GB)
```

교시별로 준비됐는지(✓/✗)와 빠진 것만 보여 준다. 다 된 항목까지 보려면 `--all`. `--fix`가 대신 돌리는 받기 명령은 아래와 같다. 로그인은 필요 없다.

```bash
uv run python scripts/data_version.py pull v1          # 코퍼스 · 질문 · 분할
uv run python scripts/download_model_hf.py             # 학습 전 e5 (HF가 막히면 download_model_gdrive.py)
uv run ragkit export-onnx models/multilingual-e5-small # ONNX 추론용 변환
uv run python scripts/finetuned_drive.py download multilingual-e5-small 002_hard_negatives 001_baseline optimize   # 학습 전 e5 인덱스 · 파인튜닝 모델 2개 · 7교시 INT8 · 가지치기 모델 (각 인덱스 포함)
```

`GEMINI_API_KEY`는 없어도 된다. LLM 답변 단계만 건너뛴다.

## 2. 완성품 써 보기 (한 번에 보기)

같은 질문 넷을 학습 전 모델과 파인튜닝 모델에 한 번에 던져 비교하고, test 전체 지표와 완성품(웹 · CLI · MCP) 띄우는 안내를 본다.

```bash
uv run python lecture/03_setup/02_try_product.py
```

## 3. CLI로 직접 질문 넷을 학습 전 모델에 던져 본다

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

## 4. 학습 전후 비교: 모델만 바꾼다

같은 명령 뒤에 `--checkpoint models/finetuned/002_hard_negatives --backend torch`만 붙이면 파인튜닝한 모델(002_hard_negatives)로 찾는다. 인덱스는 자동으로 그 모델 것을 쓴다.

```bash
uv run --package ragkit-search ragkit-search search "편의점 알바 3개월 했는데 주휴수당 받을 수 있나요?" --checkpoint models/finetuned/002_hard_negatives --backend torch
uv run --package ragkit-search ragkit-search search "회사 그만뒀는데 퇴직금은 언제까지 받아야 해요?" --checkpoint models/finetuned/002_hard_negatives --backend torch
uv run --package ragkit-search ragkit-search search "방문판매로 산 정수기 환불하고 싶어요" --checkpoint models/finetuned/002_hard_negatives --backend torch
```

말투만 바꿔서도 비교해 본다. 학습 전 모델은 법률 용어로 물어야 찾고, 파인튜닝한 모델은 평소 말투로도 찾는다.

```bash
uv run --package ragkit-search ragkit-search search "주휴일 유급휴일 1주 개근"                        # 학습 전 · 법률 용어
uv run --package ragkit-search ragkit-search search "편의점 알바를 3개월 했는데 주휴수당을 받을 수 있나요?"        # 학습 전 · 일상어
uv run --package ragkit-search ragkit-search search "편의점 알바를 3개월 했는데 주휴수당을 받을 수 있나요?" --checkpoint models/finetuned/002_hard_negatives --backend torch   # 파인튜닝 · 일상어
uv run --package ragkit-search ragkit-search search "주휴일 유급휴일 1주 개근" --checkpoint models/finetuned/002_hard_negatives --backend torch   # 파인튜닝 · 법률 용어
```

정답 근로기준법 제55조의 순위(문서 단위, 강사 PC): 학습 전 일상어 82위 · 법률 용어 3위 → 파인튜닝 일상어 2위 · 법률 용어 2위.

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
| `01_doctor.py` | 3~8교시 준비 상태를 교시별로 점검, 받은 모델을 열어 검색해 보는 동작 확인, `--fix`로 한 번에 받기 (`--quick`이면 동작 확인 생략, `--all`이면 전체 항목) |
| `02_try_product.py` | 위 질문 넷을 학습 전 · 파인튜닝 모델에 한 번에 던져 top-3와 정답 순위를 나란히 보여 준다. 웹 · CLI · MCP 띄우는 명령 안내 |
| `extra_gemini_basics.py` | (선택) Gemini API 기초. `GEMINI_API_KEY`가 있어야 하고 한 번에 4회 호출한다 |

웹(법령 노트)과 MCP는 8교시(`lecture/08_product/`)에서 직접 띄운다.
