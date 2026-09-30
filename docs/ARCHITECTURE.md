# 아키텍처

경계: 모델·데이터를 **만드는** 것은 `ragkit`(1단계 라이브러리), 만든 것을 **쓰는** 것은 `apps/*`.
앱은 ragkit의 공개 함수와 `ragkit.service.Searcher`만 쓴다.

## 데이터 흐름

```
[준비 · 1단계]
legalize-kr 법령 ─ scripts/prepare_law_data.py ─→ data/processed/law_docs.json (조문 조각 약 2.6만)
Claude skill(law-question-gen) ─→ data/questions/*.jsonl ─ ragkit split ─→ data/splits/{train,dev,test}.jsonl
                                                         ─ ragkit train ─→ models/finetuned/<실험>/

[인덱스]
law_docs.json ─ ragkit index ─→ data/processed/index/<모델 키>.sqlite
                                  docs(원문·메타데이터) + vec_docs(sqlite-vec, cosine) + meta

[최적화 · 3단계]
models/<모델> ─ ragkit export-onnx ─→ onnx/model.onnx ─ ragkit quantize ─→ models/<모델>-int8/
        └ ragkit prune-vocab ─→ models/<모델>-pruned/ ─ export-onnx ─ quantize ─→ models/<모델>-pruned-int8/ (약 30MB)

[서비스 · 2·4단계]
질문 → Searcher.search / answer_stream
        ├ embed_fn(profile.format_query(질문))   onnx | torch | st 백엔드
        ├ VectorIndex.search(k, where=법령 필터)
        └ Gemini(조문만 근거, [n] 인용) → SSE: hits → delta… → done
   ↑ apps/api (FastAPI) ← apps/web (React)
   ↑ apps/search-cli (터미널) · apps/mcp (Claude 등 에이전트)
```

## ragkit (`src/ragkit/`)

| 모듈 | 역할 |
|---|---|
| `config.py` | Settings(.env). `RAGKIT_PROJECT_ROOT`(없으면 cwd) 기준으로 data/·models/ 경로 |
| `data` | 코퍼스·질문 로드, `doc_text`, `relevance_key`(parent_id), 질문 필터 |
| `embeddings` | `create_embedding_fn(model, backend=...)`, 모델 프로필(`get_profile`: 모델별 쿼리/문서 형식·백엔드), 길이순 배치 |
| `embeddings/onnx_backend` | tokenizers + onnxruntime + numpy (배포 기본, torch 불필요) |
| `embeddings/torch_backend` | transformers + 평균 풀링 (extra `[torch]`, 장치 auto: cuda → mps → cpu) |
| `embeddings/st_backend` | sentence-transformers (EmbeddingGemma처럼 자체 풀링·Dense가 있는 모델, extra `[train]`) |
| `models` | 모델 로더, Gemini 클라이언트(토큰 사용량·재시도·스트리밍), `export_onnx`, `quantize_onnx`(채널별 INT8), `prune_vocab`(어휘 가지치기) |
| `retrieval` | `build_index`/`VectorIndex`(SQLite + sqlite-vec, 메타데이터 필터), `model_key`, `default_index_path` |
| `rag` | 조문 근거 프롬프트, 순수 LLM / RAG 답변 (1단계 비교 실습) |
| `service` | `Searcher`: 앱들의 입구. 검색·조문 조회·법령 목록·스트리밍 답변 |
| `training` | 법령 단위 분할, 대조 학습 예시, sentence-transformers 학습(전체 / LoRA) |
| `evaluation` | 전체 코퍼스 대상 Recall@k·MRR·nDCG (doc / article 판정) |
| `cli` | `ragkit`: index · split · train · evaluate · compare · export-onnx · quantize · prune-vocab |
| `retrieval.document_store`, `rag.rag_agent`, `monitoring` | 이전 구성 자료(`tutorials/_legacy`)용. 새 코드는 쓰지 않는다 |

의존성: core(onnxruntime, tokenizers, sqlite-vec, numpy, google-genai, cyclopts…) / extra `[torch]` / extra `[train]`.

## 앱 (`apps/`)

| 앱 | 단계 | 구성 |
|---|---|---|
| `api` (ragkit-api) | 2 | FastAPI. 시작 시 Searcher 로드(lifespan), 검색·답변(SSE)·노트북 저장(SQLite), `--web-dist`로 화면 제공 |
| `bench` (ragkit-bench) | 3 | 원본/ONNX/INT8을 변형마다 새 프로세스에서 측정 → 비교표 |
| `search-cli` (ragkit-search) | 4 | 서버 없이 Searcher 직접 사용. ragkit core만 의존 → uvx 배포 |
| `mcp` (ragkit-mcp) | 4 | 공식 mcp SDK(stdio). 도구: search_laws · get_article · list_laws · ask |
| `web` | 4 | Vite + React + TS + Tailwind + shadcn/ui + TanStack Query. api만 호출 |

## 실측 요약 (Mac, e5-small, test 질문 443개, 2026-09)

| | 값 |
|---|---|
| 베이스 모델 R@5 | e5-small 0.535 / EmbeddingGemma-300m 0.819 |
| 배포 변형 (torch fp32 → ONNX INT8 → 가지치기+INT8) | 모델 471 → 118 → 30 MB, 설치 669 → 127 MB, 메모리 1000 → 871 → 378 MB, R@5 0.535 → 0.535 → 0.535 |
| 인덱싱 (전체 코퍼스) | e5 mps + 길이순 배치 약 1.3분 / EmbeddingGemma 약 16분 |
