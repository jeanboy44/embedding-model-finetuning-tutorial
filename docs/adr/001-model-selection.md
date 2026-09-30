# 001: 모델 선택 - multilingual-e5-small

## 배경

교육 목적으로 임베딩 모델을 선택할 때 다음 조건이 필요했습니다:
- 학생 컴퓨터(CPU)에서 실행할 수 있을 정도로 가벼운 모델
- 실습 데이터(`data/sample_docs.json`)가 한국어이므로 한국어 지원
- 라이선스 동의나 계정 없이 받을 수 있는 공개 모델

## 결정

[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)을 기본 임베딩 모델로 선택했습니다.

## 검토한 대안

| 모델 | 파라미터 | 차원 | 한국어 | 라이선스 | 탈락 이유 |
|---|---|---|---|---|---|
| **multilingual-e5-small** | 118M | 384 | 지원 | MIT | - |
| thenlper/gte-small | 33M | 384 | 약함 (영어 중심) | MIT | 한국어 실습 데이터에서 품질이 낮음 |
| google/embeddinggemma-300m | 300M | 768 | 지원 | Gemma Terms (HF 동의 필요) | 용량(1.2GB)과 라이선스 동의 절차가 학생에게 부담 |

## 이유

1. **한국어 지원**: 다국어 학습 모델이라 한국어 문서 검색이 가능
2. **모델 크기**: 118M 파라미터(약 470MB)로 CPU 추론 가능
3. **호환성**: 384차원이라 기존 코드와 저장된 인덱스 구조를 그대로 사용
4. **가용성**: MIT 라이선스로 동의 없이 받을 수 있고, Google Drive로 재배포도 가능

## 결과

- 모델은 `scripts/download_model_hf.py` 또는 `scripts/download_model_gdrive.py`로 `models/`에 받는다.
- e5 계열은 입력 앞에 역할 문구를 붙여야 한다: 쿼리는 `query: `, 문서는 `passage: `.
  모델별 입력 형식은 `ragkit.embeddings.profiles`(모델 프로필)가 정한다. e5 계열 앞 문구는 `.env`의 `QUERY_PREFIX`, `PASSAGE_PREFIX`로 바꿀 수 있다.
- 문장 임베딩은 패딩을 제외한 mean pooling 후 L2 정규화한다 (e5 공식 방식).

## 베이스 모델 비교 대상 (2026-09-29 추가)

파인튜닝 대상(기본 모델)은 그대로 `multilingual-e5-small`이다.
1단계에서 "파인튜닝한 작은 모델이 더 크고 새로운 베이스 모델과 비교해 어떤가"를 보이기 위해, 아래 모델을 같은 평가셋(전체 코퍼스, Recall@k·MRR·nDCG)으로 비교한다.

| 모델 | 파라미터 | 차원 | 최대 토큰 | 라이선스 | 입력 형식 |
|---|---|---|---|---|---|
| multilingual-e5-small (기준) | 118M | 384 | 512 | MIT | `query: ` / `passage: ` 앞 문구 |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | 300M | 768 (MRL 512·256·128) | 2,048 | Gemma (HF 동의 필요, gated) | 쿼리 `task: search result \| query: `, 문서 `title: none \| text: ` |

주의할 점:
- 모델마다 쿼리·문서 앞 문구가 다르다. 모델 프로필(`get_profile(model).format_query / format_doc`)로 처리한다.
- EmbeddingGemma 문서 앞 문구는 `title: {제목} | text: ` 형식이고 제목이 없으면 `none`을 쓴다. 조문은 제목(`title`)이 있으니 넣는 편이 성능에 유리하다.
- EmbeddingGemma는 float16을 지원하지 않는다. float32나 bfloat16으로 돌린다. 받으려면 HF 계정으로 라이선스에 동의해야 한다.
- EmbeddingGemma는 e5-small보다 커서 CPU에서 전체 코퍼스(약 2.6만 문서) 임베딩 시간이 길다. 강의에서는 미리 만든 인덱스나 결과를 제공하는 방안을 검토한다.
- EmbeddingGemma까지 파인튜닝할지는 정하지 않았다.
