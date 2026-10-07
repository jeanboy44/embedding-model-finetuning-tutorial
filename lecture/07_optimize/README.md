# 실습 4 (7교시): 서빙과 배포 최적화

장표: [7교시 실습 4 배포 최적화](https://claude.ai/artifact/F1mKY7iVNVaKZj5psadPvR). API 복습 → ONNX 변환 · INT8 양자화 · 어휘 가지치기 → 비교표 → API에 적용 순서로 본다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.
API 키는 없어도 된다(있으면 1번에서 LLM을 1회 부른다).

## 1. API 복습

6교시까지 만든 모델을 검색 API로 감싼 것을 다시 본다: 시작할 때 모델 로딩, health, 검색, 법령 필터, 잘못된 입력(422), 스트리밍 답변.

```shell
uv run python lecture/07_optimize/01_api_review.py
```

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

만든 모델을 비교한다: torch vs ONNX, 채널별 vs 텐서 단위 INT8(test R@5 0.514 vs 0.458, 실험 012), 가지치기 후 질문 토큰화가 원본과 같은지.

```shell
uv run python lecture/07_optimize/02_onnx_quantize_prune.py
```

선택: 같은 변환을 임시 폴더에서 처음부터 직접 해 보고 텐서 단위 INT8의 코사인까지 잰다(약 30초).

```shell
uv run python lecture/07_optimize/02_onnx_quantize_prune.py --run
```

## 3. 비교표와 API에 적용

새 모델로 검색하려면 그 모델로 만든 인덱스가 필요하다. 모델마다 코퍼스 2.6만 문서를 임베딩하므로 몇 분씩 걸린다(쉬는 시간 전에 걸어 둔다).

```shell
uv run ragkit index --model models/multilingual-e5-small-int8
uv run ragkit index --model models/multilingual-e5-small-pruned-int8
```

원본 / ONNX / INT8 / 가지치기+INT8의 크기 · 메모리 · 로딩 · 지연 · 정확도 비교표(실험 008)를 읽고, API 백엔드를 torch → ONNX로 바꿔 시작 시간과 검색 속도를 잰다.

```shell
uv run python lecture/07_optimize/03_bench_and_serve.py
```

설치 669 → 127MB, 모델 471 → 30MB, 메모리 1,133 → 403MB, R@5 0.513 → 0.515. API 시작 약 3.3 → 0.1초, 검색 약 15 → 9ms.

선택: test 앞 200개로 `ragkit-bench`를 직접 돌린다(약 1분, 결과는 임시 폴더).

```shell
uv run python lecture/07_optimize/03_bench_and_serve.py --run
```

가장 작은 모델로 실제 API 서버를 띄워 본다(Ctrl+C로 끝). 브라우저에서 http://127.0.0.1:8000/docs.

```shell
uv run --package ragkit-api ragkit-api --model models/multilingual-e5-small-pruned-int8
```

8000번 포트를 다른 프로그램이 쓰고 있으면 `--port 8010`처럼 바꾼다.

## 선택: 임베딩 속도 옵션

장치(CPU·GPU), 길이순 배치, 스레드 수에 따라 인덱싱 속도가 어떻게 달라지는지 잰다(약 2분).

```shell
uv run python lecture/07_optimize/extra_embedding_speed.py
```

## 참고

| 파일 | 내용 |
|---|---|
| `01_api_review.py` | 검색 API 짧은 복습 (apps/api, FastAPI) |
| `02_onnx_quantize_prune.py` | ONNX 변환, INT8 텐서 단위 vs 채널별, 어휘 가지치기, 받은 모델 비교 |
| `03_bench_and_serve.py` | 비교표(실험 008) 읽기, API 백엔드 교체 전후 측정 |
| `extra_embedding_speed.py` | (선택) 인덱싱 속도 옵션 비교 |
