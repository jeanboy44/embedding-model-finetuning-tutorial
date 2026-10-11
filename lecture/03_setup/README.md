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
오늘 만드는 파인튜닝 모델(정답이 10위 안에 드는 비율 63% → 85%)을 7교시 방법(ONNX · INT8 · 어휘 가지치기)으로 30MB까지 줄여
조문 데이터와 함께 파일 하나에 넣은 것이다. 만드는 법은 [`apps/offline-search/README.md`](../../apps/offline-search/README.md).

가장 먼저, 저장소를 받기도 전에 이 파일부터 받는다. 아래 링크에서 내려받아 더블클릭한다.

**받기: [law-search-offline.html (Google Drive)](https://drive.google.com/file/d/1TzgE1mh2ZSyAjmPFJtb7OTuk05JFr3mt/view?usp=sharing)**

## 1. 환경 점검

```bash
uv sync --all-packages --all-extras
uv run python lecture/03_setup/01_doctor.py --fix   # 3~8교시에 쓸 것을 모두 받고 다시 점검 (처음 한 번, 약 1.5GB)
```

`01_doctor.py`는 3~8교시 실습에 필요한 것을 네 묶음으로 점검한다. 모델을 여는 동작 확인까지 수십 초, `--quick`이면 몇 초다.

1. **설치.** Python 3.10 이상, `ragkit`(ONNX 추론), extra `[torch]` · `[train]` · `[mlflow]`, 앱 네 개(`ragkit-search` · `ragkit-api` · `ragkit-bench` · `ragkit-mcp`), 디스크 여유 5GB, Node.js + pnpm을 본다. 패키지가 빠지면 받는 명령은 모두 `uv sync --all-packages --all-extras` 하나다.
2. **모델 · 인덱스.** 학습 전 e5(`models/multilingual-e5-small`)와 ONNX 변환본, 그 인덱스 `data/processed/index/multilingual-e5-small.sqlite`, 파인튜닝 모델 둘(`002_hard_negatives`는 최종 모델, `001_baseline`은 5 · 6교시)과 각 인덱스, 6교시의 실험 001 test 결과 JSON, 7교시의 INT8 · 가지치기 모델 3개와 인덱스 2개를 본다. 인덱스는 만드는 데 몇 분이라 Drive에서 받는다. 받아 둔 파인튜닝 모델이 Drive에 새로 올린 것과 다르면 "옛 버전"으로 표시하고 `--fix`가 새로 받는다.
3. **데이터.** 코퍼스 `data/processed/law_docs.json`, 질문 폴더 `data/questions/`, 법령 단위 분할 `data/splits/`(train · dev · test), 조 단위 분할 `data/splits_article/`를 보고 질문 수를 함께 찍는다. 받은 데이터가 버전 v1과 같은지도 보는데, 이것은 선택 항목이라 내가 만든 질문 · 분할이 있어 달라도 된다.
4. **동작 확인.** `ragkit.service.Searcher.open`으로 학습 전 e5와 `002_hard_negatives`를 실제로 열어 "야간 근로 수당"을 검색한다. 결과가 나오고 인덱스의 문서 수가 코퍼스 조각 수 25,967개와 같아야 통과다. 파일이 있어도 깨졌거나 코퍼스와 인덱스 버전이 다른 경우를 여기서 잡는다. `--quick`은 이 단계를 건너뛴다.

출력은 "교시별 준비" 여섯 줄(3~8교시, ✓ 준비됨 / ✗ 빠진 것 n개)이 먼저 나온다. 빠진 것이 있으면 그 아래 "빠진 것" 표(항목 · 필요한 교시 · 설명)와 "받는 법"(`--fix`가 돌릴 명령, 손으로 할 일)이 붙고 종료 코드가 1이다. 다 됐으면 "모두 준비됐습니다"와 다음 스크립트를 안내한다. 없어도 실습이 되는 선택 항목은 △로 따로 보여 준다. 다 된 항목까지 그룹별로 보려면 `--all`.

`--fix`는 빠진 항목의 받는 명령만 모아 설치 → 데이터 → 학습 전 모델 → ONNX 변환 → Drive 순서로 돌리고, 끝나면 다시 점검한다. 이미 있는 것은 다시 받지 않고, 한 명령이 실패해도 다음 명령으로 간다. 대신 돌리는 명령은 아래와 같다. 로그인은 필요 없다.

```bash
uv run python scripts/data_version.py pull v1          # 코퍼스 · 질문 · 법령 단위 분할
uv run ragkit split data/questions --unit article --test-ratio 0.15 --dev-ratio 0.1 --out data/splits_article   # 조 단위 분할 (5 · 6교시, 몇 초)
uv run python scripts/download_model_hf.py             # 학습 전 e5 (HF가 막히면 download_model_gdrive.py)
uv run ragkit export-onnx models/multilingual-e5-small # ONNX 추론용 변환
uv run python scripts/finetuned_drive.py download multilingual-e5-small 002_hard_negatives 001_baseline results_001_baseline optimize   # 학습 전 e5 인덱스 · 파인튜닝 모델 2개 · 6교시 결과 JSON · 7교시 INT8 · 가지치기 모델 (각 인덱스 포함)
```

## 2. 완성품 써 보기 (한 번에 보기)

같은 질문 넷을 학습 전 모델과 파인튜닝 모델에 한 번에 던져 비교하고, test 전체 지표와 완성품(웹 · CLI · MCP) 띄우는 안내를 본다. 받은 모델과 인덱스로 검색만 하므로 `--run` 모드는 없다(강사 PC 약 20초, 처음 실행은 40초 남짓).

```bash
uv run python lecture/03_setup/02_try_product.py
```

1. **모델 두 개 열기.** 앱(API · CLI · MCP)과 같은 입구 `ragkit.service.Searcher.open`으로 학습 전 e5(ONNX)와 `models/finetuned/002_hard_negatives`(torch, 받은 모델에는 ONNX 변환본이 없다)를 연다. 두 줄 모두 인덱스 조각 25,967개로 같고, 인덱스 파일은 모델마다 따로다. 같은 조문도 모델이 다르면 벡터가 다르기 때문이다. 여는 데 강사 PC에서 약 11초다(처음 실행은 20초 남짓). 파인튜닝 줄에는 폴더 이름(`002_hard_negatives`)을 찍는다.
2. **같은 질문을 두 모델로.** 아래 3절의 질문 넷을 두 모델로 `Searcher.search` 해 정답 조문의 순위(100위까지, 조 단위)와 top-3를 나란히 찍는다. ★가 정답 조문이다. 점수 크기는 모델마다 달라 비교하지 않고 순위만 본다. 키워드 "야간 근로 수당"은 둘 다 2 · 3위로 상위인데, 상황 설명 "편의점 알바 … 주휴수당"은 학습 전 65위에서 파인튜닝 2위로 뛴다. "방문판매로 산 정수기 환불"은 파인튜닝해도 16위라 못 찾는다. 이것이 6교시 오답 분석으로 이어진다.
3. **test 전체로 보면.** 검색하지 않고 강사가 실험 002를 끝내며 test로 한 번 잰 숫자를 보여 준다. test 질문 1,863개(조 단위 분할)의 R@5가 0.545 → 0.773이고, 유형별로는 keyword 0.739 → 0.875, question 0.573 → 0.795, situation 0.417 → 0.701이다. 상황 설명 질문에서 학습 전 모델이 가장 약하고 차이도 가장 크다. 전체 평가는 5 · 6교시에서 `ragkit evaluate`로 직접 한다.
4. **완성품 세 가지.** CLI(`ragkit-search search`), MCP(`claude mcp add`), 웹(`ragkit-api --web-dist apps/web-lite`)을 띄우는 명령과 8교시 스크립트를 안내만 한다. 서버는 띄우지 않는다. 세 제품 모두 같은 `Searcher`를 쓰므로 `--checkpoint models/finetuned/002_hard_negatives --backend torch`를 붙이면 파인튜닝 모델로 바뀐다. LLM 답변 단계는 이 강의에서 쓰지 않는다.

## 3. CLI로 직접 질문 넷을 학습 전 모델에 던져 본다

2절의 질문 넷을 이번에는 검색 CLI로 하나씩 던진다. `ragkit-search search`는 2절과 같은 `Searcher`를 열어 기본 모델(학습 전 e5, ONNX)로 검색하고, 상위 5개(`-k`로 바꾼다)를 순위 · 점수 · 제목 · 조 id · 본문 앞부분으로 보여 준다. 명령마다 모델을 새로 열므로 한 번에 몇 초 걸린다. 정답으로 떠야 할 조문이 위쪽에 뜨는지 본다. CLI는 조각(항 · 호) 단위로 보여 주므로 2절의 조 단위 순위와 조금 다를 수 있다. `show`는 조 id로 그 조의 조각을 모두 이어 원문을 보여 준다.

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

같은 명령 뒤에 `--checkpoint models/finetuned/002_hard_negatives --backend torch`만 붙이면 파인튜닝한 모델(002_hard_negatives)로 찾는다. 인덱스는 모델 키(폴더 이름 + 가중치 수정 시각)로 자동으로 그 모델 것을 쓴다. `--backend torch`는 받은 모델에 ONNX 변환본이 없어서 붙이고, 7교시에서 변환한 뒤에는 없어도 된다. 3절과 같은 질문을 던져 정답 조문의 순위가 어떻게 바뀌는지 본다.

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

강사 PC에서 잰 정답 순위 (조 단위, 전체 2.6만 문서 중, 2절 스크립트 출력):

| 질문 | 학습 전 | 파인튜닝 |
|---|---|---|
| 야간 근로 수당 | 2위 | 3위 |
| 편의점 알바 3개월 했는데 주휴수당 받을 수 있나요? | 65위 | 2위 |
| 회사 그만뒀는데 퇴직금은 언제까지 받아야 해요? | 5위 | 3위 |
| 방문판매로 산 정수기 환불하고 싶어요 | 100위 밖 | 16위 |
| 주휴일 유급휴일 1주 개근 (법률 용어) | 3위 | — |

네 질문은 체감용 예시다. 학습 질문이 없던 조로만 모은 test 전체의 성적(R@5 0.545 → 0.773, 실험 002)은 2절 3번에서 봤고, 그 숫자를 재는 법은 5 · 6교시에서 직접 해 본다.

## 스크립트

| 파일 | 내용 |
|---|---|
| `01_doctor.py` | 3~8교시 준비 상태를 설치 · 모델 · 데이터 · 동작 확인 네 묶음으로 점검, `--fix`로 한 번에 받기 (`--quick`이면 동작 확인 생략, `--all`이면 전체 항목) |
| `02_try_product.py` | 위 질문 넷을 학습 전 · 파인튜닝 모델에 한 번에 던져 top-3와 정답 순위를 나란히, test R@5 비교표, 웹 · CLI · MCP 띄우는 명령 안내 |

웹(법령 노트)과 MCP는 8교시(`lecture/08_product/`)에서 직접 띄운다.
