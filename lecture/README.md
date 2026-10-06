# 강의 실습 (lecture/)

8교시 강의의 실습 자료다. 교시마다 **장표 하나(`slides/`) + 그 업무를 직접 해 보는 스크립트 묶음(교시 폴더)** 이 짝이다. 교시 구성은 [`docs/PLAN.md`](../docs/PLAN.md) "교시 구성"을 따른다.

| 폴더 | 교시 | 실습 | 업무 |
|---|---|---|---|
| `00_setup/` | 3 | 실습 0 | 환경 점검, 완성품 먼저 써 보기, 왜 RAG·왜 파인튜닝인가 |
| `01_data/` | 4 | 실습 1 | 코퍼스, 질문 생성(Claude 스킬), 법령 단위 분할 |
| `02_evaluate/` | 5 | 실습 2 | 평가셋과 지표, 학습 없이 쓸 수 있는 선택지(쿼리 확장) |
| `03_train/` | 6 | 실습 3 | 학습 맛보기, 실험 비교, 오답 분석 |
| `04_optimize/` | 7 | 실습 4 | API 복습, ONNX · INT8 · 어휘 가지치기, 비교표와 서빙 |
| `05_product/` | 8 | 실습 5 | 검색 CLI, MCP, 웹, MLflow 모니터링 |
| `appendix/` | — | 선택 | 임베딩 속도 옵션, 임베딩 공간 탐색, Gemini 기초 |
| `slides/` | 3~8 | | HTML 장표 (교시마다 한 파일) |

실행은 모두 저장소 루트에서 한다.

```bash
uv run python lecture/<폴더>/<파일>.py          # 받은 산출물로 확인 (기본)
uv run python lecture/<폴더>/<파일>.py --run    # 작은 부분집합으로 직접 실행
```

## 두 가지 모드

무거운 산출물(모델, 파인튜닝 모델, 인덱스, 질문, 분할, 쿼리 확장 캐시, 실험 결과)은 강사가 미리 만들어 Drive로 나눠 준다.

- **기본: 받은 산출물로 확인.** 받은 파일을 읽어 결과를 보여 준다. 스크립트 하나가 몇 분 안에 끝난다.
- **`--run`: 작은 부분집합으로 직접 실행.** 같은 일을 질문 몇십 개, 학습 몇 step처럼 작게 직접 돌려 본다. 결과는 임시 폴더나 `experiments/*/results/` 아래 별도 이름에 쓰고, 받은 산출물을 덮어쓰지 않는다.

산출물이 없으면 스크립트는 스택트레이스 대신 무엇이 없고 어떤 명령으로 받는지 알려 주고 끝난다. 준비 상태는 `00_setup/01_check_env.py`가 한 번에 점검한다.

## 스크립트 작성 규칙

- 머리 docstring: 제목(`실습 N-k (M교시): …`) · 학습 목표 · 사전 준비 · 실행
- ragkit과 apps의 **공개 함수와 CLI만** 부른다. 실습 스크립트에 로직을 새로 만들지 않는다. 필요한 기능이 없으면 ragkit에 넣는다
- 경로는 `ROOT = Path(__file__).resolve().parents[2]` 또는 `ragkit.config.get_settings()` 기준. 상대 경로 금지
- e5 접두어(`query: ` / `passage: `)는 `ragkit.embeddings.format_queries` · `format_passages`로 붙인다
- 강의 모델은 `intfloat/multilingual-e5-small`과 그것을 파인튜닝한 모델뿐이다(EmbeddingGemma는 2026-10-06에 뺐다)
- 숫자는 실행 결과에서 읽어 출력한다. 문장에 박아 넣은 수치는 출처(실험 번호)를 함께 적는다
- Gemini는 `.env`의 `GEMINI_API_KEY`가 있을 때만 부르고, 무료 등급(하루 20회)을 생각해 한 번 실행에 몇 회 이하로 쓴다. API 오류(`google.genai.errors.APIError`)는 잡아서 안내한다
- 교시당 기본 모드 실행 시간 합계는 30분 안쪽
