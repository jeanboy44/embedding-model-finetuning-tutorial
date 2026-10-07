# 강의 실습 (lecture/)

8교시 강의 자료다. **폴더 하나 = 교시 하나**이고, 폴더 번호가 교시 번호다. 교시마다 장표 하나(아래 Slides 덱)와 그 업무를 직접 해 보는 스크립트 묶음이 짝이다. 교시 구성은 [`docs/PLAN.md`](../docs/PLAN.md) "교시 구성"을 따른다.

| 폴더 | 교시 | 구분 | 내용 |
|---|---|---|---|
| `01_introduction/` | 1 | 이론 | 내 소개: DS의 일, 2022 vs 2026. 1교시 장표를 내보낸 HTML 하나만 둔다 |
| `02_concepts/` | 2 | 이론 | 용어·개념, 왜 임베딩 파인튜닝인가 |
| `03_setup/` | 3 | 실습 0 | 환경 점검(doctor), 완성품 먼저 써 보기, CLI로 학습 전후 비교(README). 선택: Gemini 기초 |
| `04_data/` | 4 | 실습 1 | DS의 일 2022 vs 2026, 질문 생성을 손으로 vs 에이전트로. 코퍼스 · 분할 스크립트는 5교시에 |
| `05_evaluate/` | 5 | 실습 2 | 데이터(코퍼스 · 분할) 확인, 평가셋과 지표, 쿼리 확장, 기준 실험 002 한 바퀴(파일럿은 `06_train/01_train_taste.py --run`) |
| `06_train/` | 6 | 실습 3 | 우리가 세운 가설 전부, LoRA(004) 한 바퀴(`--run --lora` 파일럿 · 비교 · 검정), 오답 분석 |
| `07_optimize/` | 7 | 실습 4 | API 복습, ONNX · INT8 · 어휘 가지치기, 비교표와 서빙. 선택: 임베딩 속도 옵션 |
| `08_product/` | 8 | 실습 5 | 검색 CLI, 에이전트 스킬(CLI + SKILL.md), MCP, 웹, MLflow 모니터링 |

폴더 안의 `01_*.py`, `02_*.py` …는 수업 순서대로 실행하는 스크립트이고, `extra_*.py`는 시간이 남을 때 하는 선택 실습이다. 스크립트 제목의 "실습 N-k"는 실습 번호(0~5)다.

## 장표

장표는 교시마다 claude.ai Slides 덱 하나다(2026-10-06, .pptx·PDF로 내보낼 수 있다).

| 교시 | 장표 |
|---|---|
| 1 | [내 소개: DS의 일, 2022 vs 2026](https://claude.ai/artifact/QsNeCea9tMhtxGTpHd89jh) |
| 2 | [용어 · 왜 임베딩 파인튜닝인가 · 실습 지도](https://claude.ai/artifact/YCojzoPbMkeAamJ7WQhSEG) |
| 3 | [실습 0: 환경 · 완성품 · RAG 필요성](https://claude.ai/artifact/VuQyTKs7BHKu4Hsq5xy5Qc) |
| 4 | [DS의 일, 2022와 2026](https://claude.ai/artifact/FvbbskQyEMrxWcMJeRKka8) |
| 5 | [데이터 · 평가 · 기준 실험](https://claude.ai/artifact/EzELtdX3t269cS4DHsmX3p) |
| 6 | [실험 한 바퀴: LoRA, 그리고 딸깍](https://claude.ai/artifact/MtviA3WTikxFfG4a7YrN43) |
| 7 | [실습 4: 서빙과 배포 최적화](https://claude.ai/artifact/F1mKY7iVNVaKZj5psadPvR) |
| 8 | [실습 5: 제품화와 마무리](https://claude.ai/artifact/TgTkmnDgAnDHwvEFhUP2vE) |

## 실행

**교시마다 그 폴더의 `README.md`를 위에서부터 순서대로 따라 한다.** 명령은 모두 저장소 루트에서 실행하고, macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다(파이썬 실행은 모두 `uv run`).

```shell
uv run python lecture/<폴더>/<파일>.py          # 받은 산출물로 확인 (기본)
uv run python lecture/<폴더>/<파일>.py --run    # 작은 부분집합으로 직접 실행
```

처음 한 번:

```shell
uv sync --all-packages --all-extras
uv run python lecture/03_setup/01_doctor.py --fix
```

Windows는 처음 한 번 PowerShell에서 아래를 실행하고 **PowerShell 창을 새로 연다**. 파이썬이 파일과 출력을 UTF-8로 다루게 해서 한글 데이터가 깨지지 않게 한다.

```powershell
setx PYTHONUTF8 1
```

API 키 같은 설정은 셸마다 문법이 다른 환경 변수 대신 `.env` 파일에 둔다(`.env.example`을 복사해 `.env`로).

## 두 가지 모드

무거운 산출물(모델, 파인튜닝 모델, 인덱스, 질문, 분할, 쿼리 확장 캐시, 실험 결과)은 강사가 미리 만들어 Drive로 나눠 준다.

- **기본: 받은 산출물로 확인.** 받은 파일을 읽어 결과를 보여 준다. 스크립트 하나가 몇 분 안에 끝난다.
- **`--run`: 작은 부분집합으로 직접 실행.** 같은 일을 질문 몇십 개, 학습 몇 step처럼 작게 직접 돌려 본다. 결과는 임시 폴더나 `experiments/*/results/` 아래 별도 이름에 쓰고, 받은 산출물을 덮어쓰지 않는다.

산출물이 없으면 스크립트는 스택트레이스 대신 무엇이 없고 어떤 명령으로 받는지 알려 주고 끝난다. 준비 상태는 `03_setup/01_doctor.py`가 한 번에 점검한다.

## 스크립트 작성 규칙

- 머리 docstring: 제목(`실습 N-k (M교시): …`) · 학습 목표 · 사전 준비 · 실행
- ragkit과 apps의 **공개 함수와 CLI만** 부른다. 실습 스크립트에 로직을 새로 만들지 않는다. 필요한 기능이 없으면 ragkit에 넣는다
- 경로는 `ROOT = Path(__file__).resolve().parents[2]` 또는 `ragkit.config.get_settings()` 기준. 상대 경로 금지
- e5 접두어(`query: ` / `passage: `)는 `ragkit.embeddings.format_queries` · `format_passages`로 붙인다
- 강의 모델은 `intfloat/multilingual-e5-small`과 그것을 파인튜닝한 모델뿐이다(EmbeddingGemma는 2026-10-06에 뺐다)
- 숫자는 실행 결과에서 읽어 출력한다. 문장에 박아 넣은 수치는 출처(실험 번호)를 함께 적는다
- Gemini는 `.env`의 `GEMINI_API_KEY`가 있을 때만 부르고, 무료 등급(하루 20회)을 생각해 한 번 실행에 몇 회 이하로 쓴다. API 오류(`google.genai.errors.APIError`)는 잡아서 안내한다
- 교시당 기본 모드 실행 시간 합계는 30분 안쪽
