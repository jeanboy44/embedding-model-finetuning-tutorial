"""
Phase 3-1: 검색 파이프라인 구축
==================================

학습 목표:
- 문서 인덱싱 → 쿼리 임베딩 → 유사도 검색의 전체 파이프라인을 구축한다
- 검색 품질을 Precision, Recall, MRR로 평가한다
- 검색 파라미터(k, threshold)를 튜닝한다

실행:
    uv run python tutorials/phase3_agent/01_retrieval.py
"""

import json
import time
from pathlib import Path

from ragkit.config import get_settings
from ragkit.embeddings import create_embedding_fn, format_passages, format_queries
from ragkit.retrieval import DocumentStore, retrieve

# ============================================================
# 1단계: 검색 파이프라인 이해
# ============================================================
#
# 검색 파이프라인의 3단계:
#
#   [오프라인] 문서 수집 → 임베딩 생성 → 인덱스 저장
#   [온라인]  쿼리 입력 → 쿼리 임베딩 → 유사도 계산 → 상위 K개 반환
#
# 오프라인 단계는 한 번만 실행하고, 온라인 단계는 매 쿼리마다 실행된다.


def step1_build_pipeline() -> tuple[DocumentStore, list[dict]]:
    """검색 파이프라인을 처음부터 구축한다.

    Returns:
        (인덱싱된 저장소, 원본 문서 리스트) 튜플.
    """
    print("=" * 60)
    print("1단계: 검색 파이프라인 구축")
    print("=" * 60)

    # 1. 문서 로드
    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)
    print(f"\n  문서 로드: {len(docs)}개")

    # 2. 임베딩 생성
    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)

    texts = [d["text"] for d in docs]
    start = time.time()
    embeddings = embed_fn(format_passages(texts))
    elapsed = time.time() - start
    print(f"  임베딩 생성: {embeddings.shape} ({elapsed:.2f}초)")

    # 3. 인덱스 구축
    store = DocumentStore()
    store.add_documents(texts, embeddings)
    print("  인덱스 구축 완료")

    # 4. 검색 테스트
    query = "임베딩이란 무엇인가?"
    start = time.time()
    query_emb = embed_fn(format_queries([query]))
    results = retrieve(store, query_emb, k=3)
    search_time = (time.time() - start) * 1000

    print(f"\n  테스트 쿼리: '{query}'")
    print(f"  검색 시간: {search_time:.1f}ms")
    for i, (doc, score) in enumerate(results, 1):
        print(f"    {i}. [{score:.4f}] {doc[:60]}...")

    return store, docs


# ============================================================
# 2단계: 검색 품질 평가
# ============================================================


EVAL_SET = [
    {"query": "임베딩이란?", "relevant": ["emb_001", "dl_003"]},
    {"query": "RAG의 동작 원리", "relevant": ["rag_001", "rag_002"]},
    {"query": "프롬프트 엔지니어링 방법", "relevant": ["llm_001"]},
    {"query": "딥러닝 기초", "relevant": ["dl_001", "dl_002"]},
    {"query": "ML 시스템 모니터링", "relevant": ["mon_001", "mon_002"]},
]


def step2_evaluate(store: DocumentStore, docs: list[dict]) -> None:
    """검색 품질을 정량적으로 평가한다.

    Args:
        store: 인덱싱된 문서 저장소.
        docs: 원본 문서 리스트.
    """
    print("\n" + "=" * 60)
    print("2단계: 검색 품질 평가")
    print("=" * 60)

    print("""
주요 평가 지표:

  Precision@K = (관련 문서 수 in 상위 K) / K
    → "검색 결과 중 실제로 관련 있는 비율"

  Recall@K = (관련 문서 수 in 상위 K) / (전체 관련 문서 수)
    → "관련 문서를 얼마나 빠짐없이 찾았는가"

  MRR (Mean Reciprocal Rank) = 1/rank_of_first_relevant
    → "관련 문서가 얼마나 상위에 나오는가"
""")

    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)
    doc_id_map = {d["id"]: i for i, d in enumerate(docs)}

    for k in [1, 3, 5]:
        total_p, total_r, total_mrr = 0.0, 0.0, 0.0

        for item in EVAL_SET:
            query_emb = embed_fn(format_queries([item["query"]]))
            results = retrieve(store, query_emb, k=k)

            relevant_texts = {
                docs[doc_id_map[rid]]["text"]
                for rid in item["relevant"]
                if rid in doc_id_map
            }
            retrieved_texts = [text for text, _ in results]

            hits = sum(1 for t in retrieved_texts if t in relevant_texts)
            total_p += hits / k
            total_r += hits / len(relevant_texts) if relevant_texts else 0

            for rank, (text, _) in enumerate(results, 1):
                if text in relevant_texts:
                    total_mrr += 1.0 / rank
                    break

        n = len(EVAL_SET)
        print(
            f"  K={k}: P@{k}={total_p / n:.3f}  R@{k}={total_r / n:.3f}  MRR={total_mrr / n:.3f}"
        )


# ============================================================
# 3단계: 파라미터 튜닝
# ============================================================


def step3_parameter_tuning(store: DocumentStore, docs: list[dict]) -> None:
    """검색 파라미터의 영향을 분석한다.

    Args:
        store: 인덱싱된 문서 저장소.
        docs: 원본 문서 리스트.
    """
    print("\n" + "=" * 60)
    print("3단계: 파라미터 튜닝")
    print("=" * 60)

    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)

    query = "임베딩 모델을 파인튜닝하는 방법"
    query_emb = embed_fn(format_queries([query]))

    # K 값에 따른 결과 비교
    print(f"\n쿼리: '{query}'")
    print("\nK 값에 따른 결과:")

    for k in [1, 3, 5, 10]:
        results = retrieve(store, query_emb, k=k)
        scores = [s for _, s in results]
        print(f"\n  K={k}: 점수 범위 [{min(scores):.4f} ~ {max(scores):.4f}]")
        for i, (doc, score) in enumerate(results, 1):
            marker = "  " if score < 0.5 else ""
            print(f"    {i}. [{score:.4f}]{marker} {doc[:50]}...")

    # 유사도 임계값 분석
    print("\n유사도 임계값 분석:")
    results_all = retrieve(store, query_emb, k=len(docs))
    thresholds = [0.3, 0.5, 0.7, 0.8]
    for threshold in thresholds:
        above = sum(1 for _, s in results_all if s >= threshold)
        print(f"  threshold={threshold}: {above}개 문서 통과")

    print("""
팁:
- K가 클수록 Recall은 높아지지만, 노이즈도 증가
- 유사도 임계값으로 저품질 결과를 필터링할 수 있음
- RAG에서는 보통 K=3~5, threshold=0.5 정도가 적절
""")


# ============================================================
# 4단계: 검색 실패 분석
# ============================================================


def step4_failure_analysis(store: DocumentStore, docs: list[dict]) -> None:
    """검색 실패 사례를 분석하고 개선 방향을 제시한다.

    Args:
        store: 인덱싱된 문서 저장소.
        docs: 원본 문서 리스트.
    """
    print("\n" + "=" * 60)
    print("4단계: 검색 실패 분석")
    print("=" * 60)

    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)

    # 검색이 어려운 쿼리들
    hard_queries = [
        ("약어 사용", "ML 모니터링 어떻게 해?"),
        ("간접적 표현", "모델이 거짓말하는 것을 막으려면?"),  # → hallucination
        ("영어 혼용", "embedding fine-tuning 방법"),
        ("모호한 질문", "성능을 높이려면?"),
    ]

    for category, query in hard_queries:
        query_emb = embed_fn(format_queries([query]))
        results = retrieve(store, query_emb, k=3)

        print(f"\n  [{category}] '{query}'")
        for i, (doc, score) in enumerate(results, 1):
            print(f"    {i}. [{score:.4f}] {doc[:55]}...")

    print("""
검색 실패의 주요 원인:
1. 어휘 불일치 — "환각"과 "hallucination"을 연결 못함
2. 간접적 표현 — "거짓말 방지" → "환각 감소" 연결 어려움
3. 모호한 쿼리 — "성능"이 무엇인지 불분명

개선 방법:
1. 쿼리 확장 (Phase 1에서 학습)
2. 동의어/약어 사전 구축
3. 하이브리드 검색 (벡터 + 키워드)
4. 임베딩 모델 파인튜닝
""")


# ============================================================
# 실행
# ============================================================


def main() -> None:
    """검색 파이프라인 튜토리얼을 실행한다."""
    store, docs = step1_build_pipeline()
    step2_evaluate(store, docs)
    step3_parameter_tuning(store, docs)
    step4_failure_analysis(store, docs)

    print("=" * 60)
    print("실습 과제")
    print("=" * 60)
    print("""
1. EVAL_SET에 새로운 쿼리-정답 쌍을 3개 추가하고 평가하세요.
2. 유사도 임계값 필터링을 retrieve 함수에 추가해보세요.
3. data/sample_docs.json에 영어 문서를 추가하면 검색 품질이 어떻게 변하나요?
4. (심화) 하이브리드 검색: 벡터 유사도 + 키워드 매칭을 조합해보세요.
""")


if __name__ == "__main__":
    main()
