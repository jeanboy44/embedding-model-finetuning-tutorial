# 실습 3 (7교시): 서빙과 배포 최적화

장표: [7교시 실습 3 배포 최적화](https://claude.ai/artifact/F1mKY7iVNVaKZj5psadPvR). API 복습 → ONNX 변환 · INT8 양자화 · 어휘 가지치기 → 비교표 → API에 적용 순서로 본다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.
LLM 답변 단계는 이 강의에서 쓰지 않는다. 1번의 스트리밍은 근거 조문까지만 받고, 답 조각 대신 오류 안내가 온다.

## 0. 준비 확인

이 교시는 Drive에서 받은 산출물 네 묶음을 쓴다. `models/`의 네 폴더(`multilingual-e5-small`, `-int8`, `-pruned`, `-pruned-int8`), `data/processed/index/`의 인덱스, `data/splits/test.jsonl`, `experiments/101_deploy_bench/results/`다. 빠진 것이 있으면 스크립트가 무엇이 없는지 적고 끝난다.

```shell
uv run python lecture/03_setup/01_doctor.py
```

## 1. API 복습

6교시까지 만든 모델을 검색 API로 감싼 것을 다시 본다: 시작할 때 모델 로딩, health, 검색, 법령 필터, 잘못된 입력(422), 스트리밍 답변.

```shell
uv run python lecture/07_optimize/01_api_review.py
```

모드는 하나뿐이고 로딩까지 10~20초다. 서버를 따로 띄우지 않고 같은 앱을 프로세스 안에서 `TestClient`로 부른다. 노트북 저장소는 임시 폴더에 만들고 끝나면 지운다.

1. 준비물 확인. `data/processed/index/*.sqlite`가 하나라도 있어야 한다. 없으면 `uv run ragkit index`로 만들라고 안내하고 끝난다.
2. 앱 시작. `ragkit_api.app.create_app`으로 앱을 만들고 `TestClient`로 연다. 이때 lifespan이 `Searcher.open(backend="torch")`로 모델과 인덱스를 한 번 로드한다. 출력에서 로딩 초를 본다. 이 Mac에서 5~13초이고 처음 실행이 느리다. 이후 요청은 이 모델을 재사용한다.
3. health와 검색. `GET /api/health`가 `model_key`(multilingual-e5-small), `doc_count`(25,967), `llm_available`을 돌려준다. `POST /api/search`로 "야간에 일하면 수당을 더 받나요?"를 `k=3`으로 검색하면 소득세법 시행령 제17조가 위에 온다. 같은 질문에 `laws=["근로기준법"]`을 붙이면 근로기준법 제56조(연장ㆍ야간 및 휴일 근로)가 1위다. 첫 검색의 ms는 워밍업이라 수백 ms이고 이후 요청은 수십 ms다. search는 `Searcher.search`를 그대로 감싼 것이라 CLI `ragkit search`와 같은 결과다.
4. 잘못된 요청. 빈 질문과 `k=999`를 보내면 pydantic 스키마가 막아 HTTP 422와 메시지 두 개("query String should have at least 1 character", "k Input should be less than or equal to 50")가 온다. `laws=["없는법"]`은 앱이 `Searcher.resolve_laws`로 법령 이름을 찾지 못해 422 "모르는 법령: 없는법"이 온다. 코드 한 줄 없이 스키마가 검증과 `/docs` 문서를 함께 만든다.
5. 스트리밍 답변. `POST /api/answer/stream`에 주택임대차보호법 안에서 "월세 계약이 끝나면 보증금은 언제 돌려받나요?"를 `k=5`로 보낸다. 응답은 `text/event-stream`이고 `hits` 이벤트로 근거 5건(제6조의3 계약갱신 요구 등)이 먼저 온다. LLM 답변을 쓰지 않으므로 그 뒤 바로 `done`이 오고 `error` 칸에 이유가 적힌다. 이벤트 순서 `['hits', 'done']` 총 2개를 본다. 답 조각 `delta`는 이 강의에서 오지 않는다.
6. 노트북 API. `POST /api/notebooks`로 법령 두 개를 묶은 노트북을 만들어 id와 소스 법령을 보고, 대화(`chat`) · 기록(`messages`) · 노트(`notes`)는 LLM 답변이 필요해 호출하지 않고 순서만 적는다. `DELETE /api/notebooks/{id}` → HTTP 204로 끝난다. 8교시 웹 화면이 이 API만 부른다.

실제 서버는 `uv run --package ragkit-api ragkit-api --backend torch`로 띄우고 http://127.0.0.1:8000/docs 에서 같은 엔드포인트를 본다. 3번에서 백엔드를 ONNX로 바꾼다.

## 2. 모델 줄이기: ONNX → INT8 → 어휘 가지치기

배포용 모델 세 개를 직접 만든다. 각 명령은 수 초~수십 초다.

```shell
uv run ragkit export-onnx models/multilingual-e5-small
uv run ragkit quantize models/multilingual-e5-small
uv run ragkit prune-vocab models/multilingual-e5-small
uv run ragkit export-onnx models/multilingual-e5-small-pruned
uv run ragkit quantize models/multilingual-e5-small-pruned
```

- `export-onnx`: torch 없이 도는 ONNX 형식 (`models/multilingual-e5-small/onnx/`). 3교시에 이미 했다면 다시 해도 같다
- `quantize`: 채널별 INT8 → `models/multilingual-e5-small-int8` (471 → 118MB)
- `prune-vocab`: 다국어 어휘 25만 개 중 한국어 법령에 필요한 약 1.9만 개만 남긴다 → `-pruned` → 변환·양자화 → `models/multilingual-e5-small-pruned-int8` (30MB)

만든 모델을 비교한다: torch vs ONNX, 채널별 vs 텐서 단위 INT8(test R@5 0.514 vs 0.458, 실험 009), 가지치기 후 질문 토큰화가 원본과 같은지.

```shell
uv run python lecture/07_optimize/02_onnx_quantize_prune.py
```

기본 모드는 받은 모델 폴더 네 개를 읽어 비교만 한다. 변환과 양자화를 다시 하지 않고 수십 초에 끝난다.

1. 준비물 확인. 원본 가중치 `model.safetensors`, `data/splits/test.jsonl`, 네 폴더의 `onnx/model.onnx`가 있어야 한다. 비교용 문장은 코퍼스에서 고르게 뽑은 문서 200개(`passage:` 접두어)와 질문 3개(`query:` 접두어)로 203문장이다.
2. ONNX 변환. `onnx/model.onnx` 470MB가 가중치 471MB와 거의 같음을 본다. 같은 203문장을 `create_embedding_fn`의 torch 백엔드와 onnx 백엔드로 임베딩해 행별 코사인을 재면 평균 1.00000, 최소 1.00000이다. 형식만 바뀌고 모델은 같다.
3. INT8 양자화. fp32 ONNX를 기준으로 채널별 INT8의 크기와 코사인을 표로 본다. 470 → 118MB, 코사인 평균 0.9994, 최소 0.9991이다. 이어서 실험 101의 test R@5를 읽어 onnx-fp32 0.513 · onnx-int8 0.514 · onnx-int8-pruned 0.515를, 실험 009에서 텐서 단위 INT8 0.458을 보여 준다. 코사인 0.99의 작은 차이가 검색에서는 R@5 5점 넘는 차이로 커진다.
4. 어휘 가지치기. `safetensors` 헤더만 열어 파라미터 117.7M 중 단어 임베딩 표가 96.0M(82%)임을 센다. 코퍼스 2.6만 문서를 토큰화하면 실제 쓰는 토큰은 5,483개 / 250,002개(2.2%)다. 받은 가지치기 모델은 어휘 250,002 → 19,282개, 가지치기+INT8 30MB, fp32 대비 코사인 평균 0.9994다. test 질문 2,526개를 `query:` 접두어를 붙여 토큰화하면 2,526개(100%)가 원본과 똑같이 쪼개진다. 코사인이 1보다 작은 것은 가지치기가 아니라 INT8 때문이다.

선택: 같은 변환을 임시 폴더에서 처음부터 직접 해 보고 텐서 단위 INT8의 코사인까지 잰다. 이 Mac에서 약 3분이다.

```shell
uv run python lecture/07_optimize/02_onnx_quantize_prune.py --run
```

`--run`은 `extra [train]`이 필요하다(`uv sync --all-packages --all-extras`). 원본 가중치만 있으면 `export_onnx` → `quantize_onnx`(텐서 단위와 채널별 둘 다) → `vocab_keep_ids` · `prune_vocab` → `export_onnx` → `quantize_onnx`를 임시 폴더에서 돌린다. CLI `ragkit export-onnx` · `quantize --no-per-channel` · `prune-vocab`과 같은 함수다. 2번 표에 "INT8 텐서 단위" 줄이 하나 늘어 118MB, 코사인 평균 0.9859, 최소 0.9799로 채널별보다 멀어진 것을 본다. 임시 폴더는 끝나면 지우고 받은 `models/`는 그대로다.

## 3. 비교표와 API에 적용

새 모델로 검색하려면 그 모델로 만든 인덱스가 필요하다. 모델마다 코퍼스 2.6만 문서를 임베딩하므로 몇 분씩 걸린다(쉬는 시간 전에 걸어 둔다).

```shell
uv run ragkit index --model models/multilingual-e5-small-int8
uv run ragkit index --model models/multilingual-e5-small-pruned-int8
```

원본 / ONNX / INT8 / 가지치기+INT8의 크기 · 메모리 · 로딩 · 지연 · 정확도 비교표(실험 101)를 읽고, API 백엔드를 torch → ONNX로 바꿔 시작 시간과 검색 속도를 잰다.

```shell
uv run python lecture/07_optimize/03_bench_and_serve.py
```

기본 모드는 받은 측정 결과를 읽고 3단계의 API 비교만 직접 잰다. 약 20초다.

1. 측정 결과 읽기. 실험 101의 `results.json`(변형별 수치)과 `comparison.md`(표)를 읽어 표를 그대로 출력한다. `ragkit-bench`가 네 변형(torch-fp32 · onnx-fp32 · onnx-int8 · onnx-int8-pruned)을 변형마다 새 프로세스에서, 같은 test 질문 2,526개와 CPU로 잰 것이다. 열은 모델 MB · 설치 MB · 로딩 s · 지연 p50/p95 ms · 문서/초 · 최대 메모리 MB · R@1/5/10 · MRR@10이다.
2. 표 읽기. torch-fp32를 기준으로 변형마다 설치 크기 · 모델 크기 · 로딩 · 지연 p50 · 메모리 · R@5를 "전 → 후 (배율)"로 풀어 쓴다. onnx-fp32는 설치 669 → 127MB(1/5.3), 로딩 3.59 → 0.77초인데 메모리는 1,133 → 1,782MB로 오히려 1.6배 크다. onnx-int8은 모델 471 → 118MB(1/4.0), 메모리 874MB(23% 줄어듦), R@5 +0.001이다. onnx-int8-pruned는 모델 30MB(1/15.9), 로딩 0.30초, 메모리 403MB(1/2.8), R@5 0.513 → 0.515(+0.002)다. 정확도를 잃지 않고 무엇을 얻었는지 직접 판단한다.
3. API에 적용. 1번과 같은 `create_app`을 torch / ONNX INT8 / 가지치기+INT8로 세 번 열어 시작 시간과 같은 검색 20회의 중앙값 ms를 잰다. 모델과 백엔드 인자만 바뀌고 코드는 같다. 이 Mac에서 torch는 시작 11.0초(처음 실행, 벤치에서는 3.6초) · 검색 14ms, ONNX INT8은 0.5초 · 11ms, 가지치기+INT8은 0.1초 · 11ms다. 세 모델 모두 1위가 근로기준법 제56조로 같다. 모델 폴더가 없으면 그 줄은 건너뛴다.

정리하면 설치 669 → 127MB, 모델 471 → 30MB, 메모리 1,133 → 403MB, R@5 0.513 → 0.515(실험 101). API 시작은 수 초 → 0.1초, 검색 중앙값은 14 → 11ms다.

선택: test 앞 200개로 `ragkit-bench`를 직접 돌린다. 결과는 임시 폴더에 쓰고 지우며 받은 결과를 덮어쓰지 않는다.

```shell
uv run python lecture/07_optimize/03_bench_and_serve.py --run
```

`--run`은 `test.jsonl` 앞 200줄을 임시 파일로 쓰고 `ragkit-bench run --questions <임시> --out-dir <임시> --n-latency 30`을 부속 프로세스로 돌린다. 변형마다 새 프로세스를 띄워 로딩 · 지연 · 최대 메모리 · R@k를 재고 `results.json`과 `comparison.md`를 쓴다. 변형별 인덱스가 모두 있으면 1분 남짓이고, 없는 변형은 인덱스를 먼저 만들어 몇 분 더 걸린다. 2 · 3단계는 이 새 결과로 같은 출력을 낸다. 전체 측정은 `uv run ragkit-bench run`이다.

가장 작은 모델로 실제 API 서버를 띄워 본다(Ctrl+C로 끝). 브라우저에서 http://127.0.0.1:8000/docs.

```shell
uv run --package ragkit-api ragkit-api --model models/multilingual-e5-small-pruned-int8
```

8000번 포트를 다른 프로그램이 쓰고 있으면 `--port 8010`처럼 바꾼다.

서버가 뜬 상태에서 **새 터미널 창을 열고** API 동작을 확인한다:

- **Windows (PowerShell)**:
  ```powershell
  curl.exe http://127.0.0.1:8000/api/health
  curl.exe --% -X POST http://127.0.0.1:8000/api/search -H "Content-Type: application/json; charset=utf-8" -d "{\"query\": \"야간 근로 수당\", \"k\": 3}"
  ```

- **macOS / Linux**:
  ```bash
  curl http://127.0.0.1:8000/api/health
  curl -X POST http://127.0.0.1:8000/api/search -H "Content-Type: application/json" -d '{"query": "야간 근로 수당", "k": 3}'
  ```

배포할 것은 ragkit core(torch 없음)와 모델 폴더(약 30MB), 그 모델로 만든 인덱스 파일(`data/processed/index/<모델 키>.sqlite`)이다.

## 선택: 임베딩 속도 옵션

장치(CPU·GPU), 길이순 배치, 스레드 수에 따라 인덱싱 속도가 어떻게 달라지는지 잰다(약 2분).

```shell
uv run python lecture/07_optimize/extra_embedding_speed.py
```

모드는 하나뿐이다. 속도를 바꿔도 임베딩은 같아야 하므로 설정마다 기준 대비 코사인을 함께 본다.

1. 표본 문서. 코퍼스 2.6만 개에서 고르게 1,000개를 뽑아 `passage:` 접두어를 붙이고 길이 중앙값과 최대, 배치 크기 64를 출력한다.
2. 설정별 측정. `CONFIGS`의 설정 8개를 차례로 돈다. torch CPU(정렬 없음 · 길이 정렬 · 길이 정렬 4스레드), torch mps(정렬 없음 · 길이 정렬), ONNX CPU(정렬 없음 · 길이 정렬), ONNX CoreML이다. 이 컴퓨터에 없는 장치는 건너뛴다. 설정마다 `create_embedding_fn`으로 임베딩 함수를 만들고 워밍업 한 배치 뒤 1,000개를 재서 걸린 초 · 코퍼스 전체 환산 분 · 첫 설정 대비 최소 코사인을 한 줄로 출력한다. 코사인은 1에 가까워야 같은 결과다.
3. 속도 배율. 처음 성공한 설정(보통 torch cpu 정렬 없음)을 x1.0으로 두고 설정별 배율을 보여 준다. 길이 정렬이 왜 빨라지는지, 코사인이 0.999 아래로 떨어진 설정이 있는지 생각해 본다.

같은 옵션은 `uv run ragkit index --backend torch --device mps --batch-size 64 --threads 8`처럼 인덱싱 명령에도 있고, `.env`의 `EMBEDDING_DEVICE` · `EMBEDDING_SORT_BY_LENGTH` · `EMBEDDING_NUM_THREADS`로 기본값을 바꿀 수 있다. 여기서 잰 숫자는 이 컴퓨터 기준이라 배포 서버에서는 다시 재야 한다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_api_review.py` | 검색 API 짧은 복습 (apps/api, FastAPI) |
| `02_onnx_quantize_prune.py` | ONNX 변환, INT8 텐서 단위 vs 채널별, 어휘 가지치기, 받은 모델 비교 |
| `03_bench_and_serve.py` | 비교표(실험 101) 읽기, API 백엔드 교체 전후 측정 |
| `extra_embedding_speed.py` | (선택) 인덱싱 속도 옵션 비교 |
