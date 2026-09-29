# 001: 모델 선택 - multilingual-e5-small

## 배경

교육 목적으로 임베딩 모델을 선택할 때 다음 조건이 필요했습니다:
- 학생 컴퓨터(CPU)에서 실행할 수 있을 정도로 가벼운 모델
- 실습 데이터(법령 조문, `data/processed/law_docs.json`)가 한국어이므로 한국어 지원
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
  `src.embeddings.format_queries` / `format_passages`를 사용한다. 다른 모델로 바꾸면 `.env`에서 `QUERY_PREFIX`, `PASSAGE_PREFIX`를 조정한다.
- 문장 임베딩은 패딩을 제외한 mean pooling 후 L2 정규화한다 (e5 공식 방식).
