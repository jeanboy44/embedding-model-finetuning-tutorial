# 오프라인 법령 검색 — HTML 파일 하나

받은 HTML 파일을 더블클릭하면 브라우저에서 법령 조문 검색이 됩니다. 인터넷, 서버, API 키가 모두 필요 없습니다.
3교시에 Python을 설치하기 전, 오늘의 최종 목표(파인튜닝 → 최적화 → 파일 하나로 배포)로 먼저 보여 줍니다.
파일(약 56MB) 안에 아래가 전부 들어 있습니다.

| 들어 있는 것 | 내용 |
|---|---|
| 임베딩 모델 | 파인튜닝 모델 r001_A를 어휘 가지치기 → ONNX → INT8로 줄인 30MB 모델 |
| 조문 벡터 | 코퍼스 25,967건을 같은 모델로 임베딩한 384차원 벡터. 행마다 배율을 둔 int8로 저장 |
| 조문 원문 | 제목, 본문, 법령명, 테마, 법령 종류, 법제처 링크 (gzip) |
| onnxruntime-web | 브라우저에서 ONNX를 돌리는 엔진(wasm) |

질문은 브라우저 안에서 `query: ` 앞 문구를 붙여 임베딩하고, 조문 벡터 전체와 코사인 유사도를 비교합니다.
순위는 조 단위입니다(같은 조의 항·호 조각은 가장 높은 것 하나로 셈). `--baseline`으로 만들면 학습 전 모델도 넣어(약 103MB)
두 모델의 결과를 좌우에 놓고, 카드마다 같은 조가 상대 모델에서 몇 위였는지(100위까지) 보여 줍니다.
토크나이저(`src/tokenizer.js`)와 평균 풀링·정규화(`src/search.js`)는 파이썬의 `tokenizers`와
`ragkit.embeddings.onnx_backend`를 옮긴 것입니다. 답변 생성(LLM)은 들어 있지 않습니다.

## 만들기

저장소 루트에서 실행합니다. 파인튜닝 모델 `models/finetuned/r001_A`가 있어야 합니다
(`uv run python scripts/finetuned_drive.py download`).

```bash
# 1. onnxruntime-web 받기 (한 번만)
pnpm --dir apps/offline-search install

# 2. 파인튜닝 모델을 30MB로 줄이기: 어휘 가지치기 → ONNX → INT8
uv run ragkit prune-vocab models/finetuned/r001_A
uv run ragkit export-onnx models/finetuned/r001_A-pruned
uv run ragkit quantize models/finetuned/r001_A-pruned

# 3. 줄인 모델로 조문 벡터 만들기 (CPU 약 5분)
uv run ragkit index --checkpoint models/finetuned/r001_A-pruned-int8 --backend onnx --out data/processed/index/r001_A-pruned-int8-onnx.sqlite
# (--baseline으로 만들 때만) 학습 전 e5의 30MB판(7교시 명령으로 만든다)으로도 한 번 더
uv run ragkit index --checkpoint models/multilingual-e5-small-pruned-int8 --backend onnx --out data/processed/index/multilingual-e5-small-pruned-int8-onnx.sqlite

# 4. HTML 만들기 → dist/law-search-offline.html
uv run python apps/offline-search/build.py
```

조문 벡터는 HTML에 넣는 바로 그 ONNX 모델로 만든 인덱스여야 합니다. 질문을 그 모델로 임베딩하기 때문입니다.
다른 모델을 넣으려면 `--model`·`--index`(파인튜닝 쪽)나 `--baseline-model`·`--baseline-index`(학습 전 쪽)를 함께 바꿉니다.
학습 전 모델과 나란히 비교하는 판은 `--baseline`을 붙입니다 (학습 전 인덱스가 필요하다).

## 성능

test 질문 2,526개, 정답 조문 기준입니다. "브라우저"는 HTML 안과 같은 엔진(onnxruntime-web wasm)으로 질문을 임베딩하고
int8로 저장한 조문 벡터와 비교한 값입니다.

| 모델 | 실행 | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|---|
| 학습 전 e5 (30MB) | 파이썬 | 0.287 | 0.515 | 0.602 | 0.383 |
| 학습 전 e5 (30MB) | 브라우저 | 0.278 | 0.497 | 0.592 | 0.374 |
| 파인튜닝 r001_A (30MB) | 파이썬 | 0.447 | 0.722 | 0.804 | 0.566 |
| 파인튜닝 r001_A (30MB) | 브라우저 | 0.443 | 0.717 | 0.802 | 0.561 |

브라우저 값이 조금 낮은 이유는 INT8 연산 방식 차이입니다. 파이썬 onnxruntime은 그래프 최적화(EXTENDED)에서
양자화 행렬곱을 하나로 합친 커널을 쓰고, wasm은 합치지 않은 계산을 씁니다. 그래서 같은 ONNX 파일이어도 질문 벡터가
조금 다릅니다(코사인 약 0.997). fp32 모델은 둘이 같은 벡터를 냅니다.

## 확인

```bash
uv run pytest tests/apps/test_offline_search.py   # JS 토크나이저가 파이썬과 같은 토큰 id를 내는지 등
```

Chrome, Safari(WebKit), Firefox에서 `file://`로 열어 불러오기와 검색이 되고 바깥으로 나가는 요청이 없는 것을 확인했습니다.

## 제약

- 열 때마다 파일 안의 데이터를 풀고 모델을 불러옵니다. 빠른 노트북에서 1초 안팎이고, 느린 PC에서는 몇 초 걸릴 수 있습니다.
- `file://`로 열면 멀티스레드 wasm을 쓸 수 없어 한 스레드로 돌립니다. 질문 한 문장은 이것으로 충분히 빠릅니다(수십 ms).
- 토크나이저 정규화는 sentencepiece 표 대신 NFKC를 씁니다. 코퍼스와 질문 전체(약 3.8만 문장)에서 파이썬과 같은 id가 나오는 것을 확인했습니다.
- "법제처에서 보기" 링크만 인터넷이 필요합니다.
