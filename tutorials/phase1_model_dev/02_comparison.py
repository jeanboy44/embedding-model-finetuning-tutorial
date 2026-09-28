"""
Phase 1-2: 임베딩 방식 비교 실험
==================================

학습 목표:
- 세 가지 임베딩 접근법의 차이를 이해한다
  1) Base 모델 (사전 학습 그대로)
  2) Fine-tuned 모델 (도메인 데이터로 추가 학습)
  3) LLM 쿼리 확장 (쿼리를 LLM으로 보강 후 검색)
- 각 방식의 검색 품질을 정량적으로 비교한다
- 어떤 상황에서 어떤 방식이 유리한지 판단한다

실행:
    uv run python tutorials/phase1_model_dev/02_comparison.py
"""

import json
import time
from pathlib import Path

import numpy as np

from src.config import get_settings
from src.embeddings import create_embedding_fn, format_passages, format_queries
from src.retrieval import DocumentStore, retrieve

# ============================================================
# 데이터 준비
# ============================================================


def load_documents() -> tuple[list[str], list[dict]]:
    """샘플 문서를 로드한다.

    Returns:
        (텍스트 리스트, 원본 문서 리스트) 튜플.
    """
    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)
    texts = [d["text"] for d in docs]
    return texts, docs


# 평가용 쿼리-정답 쌍
# 각 쿼리에 대해 가장 관련 있는 문서 ID를 지정
EVAL_QUERIES = [
    {
        "query": "임베딩이란 무엇인가?",
        "relevant_ids": ["dl_003", "emb_001"],
        "category": "embeddings",
    },
    {
        "query": "RAG 시스템은 어떻게 동작하는가?",
        "relevant_ids": ["rag_001", "rag_002"],
        "category": "rag",
    },
    {
        "query": "모델 성능을 어떻게 모니터링하는가?",
        "relevant_ids": ["mon_001", "mon_002"],
        "category": "monitoring",
    },
    {
        "query": "트랜스포머 구조를 설명해주세요",
        "relevant_ids": ["dl_002", "dl_001"],
        "category": "deep_learning",
    },
    {
        "query": "검색 품질을 높이는 방법은?",
        "relevant_ids": ["rag_003", "rag_002"],
        "category": "rag",
    },
]


# ============================================================
# 1단계: Base 모델 검색
# ============================================================


def step1_base_retrieval() -> dict[str, float]:
    """Base 모델로 검색 성능을 측정한다.

    Returns:
        평가 메트릭 딕셔너리.
    """
    print("=" * 60)
    print("1단계: Base 모델 검색")
    print("=" * 60)
    print("\n사전 학습된 모델을 그대로 사용한다.")
    print("별도의 학습 없이 바로 사용할 수 있는 것이 장점.\n")

    texts, docs = load_documents()
    doc_id_map = {d["id"]: i for i, d in enumerate(docs)}

    settings = get_settings()
    embed = create_embedding_fn(settings.embedding_model_name)

    start = time.time()
    doc_embeddings = embed(format_passages(texts))
    index_time = time.time() - start

    store = DocumentStore()
    store.add_documents(texts, doc_embeddings)

    metrics = _evaluate_retrieval(store, embed, docs, doc_id_map)
    metrics["index_time_ms"] = index_time * 1000

    print(f"\n인덱싱 시간: {metrics['index_time_ms']:.1f}ms")
    print(f"Precision@3: {metrics['precision_at_3']:.3f}")
    print(f"Recall@3:    {metrics['recall_at_3']:.3f}")
    print(f"MRR:         {metrics['mrr']:.3f}")

    return metrics


# ============================================================
# 2단계: Fine-tuned 모델 시뮬레이션
# ============================================================
# 실제 파인튜닝은 시간이 오래 걸리므로,
# 여기서는 "파인튜닝의 효과"를 시뮬레이션한다.
# 실제 프로젝트에서는 experiments/exp_002_finetuned_embedding을 참고.


def step2_finetuned_simulation() -> dict[str, float]:
    """파인튜닝 효과를 시뮬레이션하여 성능을 비교한다.

    Returns:
        평가 메트릭 딕셔너리.
    """
    print("\n" + "=" * 60)
    print("2단계: Fine-tuned 모델 (시뮬레이션)")
    print("=" * 60)
    print("\n파인튜닝이란?")
    print("- 사전 학습된 모델을 특정 도메인 데이터로 추가 학습")
    print("- 도메인 용어/패턴에 대한 이해도가 향상됨")
    print("- 대조 학습: 유사 문서는 가깝게, 비유사 문서는 멀게\n")

    texts, docs = load_documents()
    doc_id_map = {d["id"]: i for i, d in enumerate(docs)}

    settings = get_settings()
    embed = create_embedding_fn(settings.embedding_model_name)

    doc_embeddings = embed(format_passages(texts))

    # 시뮬레이션: 같은 카테고리 문서끼리 임베딩을 약간 가깝게 조정
    # (실제 파인튜닝의 효과를 간단히 흉내낸다)
    category_centroids: dict[str, np.ndarray] = {}
    for i, doc in enumerate(docs):
        cat = doc["category"]
        if cat not in category_centroids:
            category_centroids[cat] = []
        category_centroids[cat].append(doc_embeddings[i])

    for cat, vecs in category_centroids.items():
        category_centroids[cat] = np.mean(vecs, axis=0)

    # 카테고리 중심 방향으로 10% 이동 (파인튜닝 효과 시뮬레이션)
    finetuned_embeddings = doc_embeddings.copy()
    for i, doc in enumerate(docs):
        centroid = category_centroids[doc["category"]]
        finetuned_embeddings[i] = 0.9 * doc_embeddings[i] + 0.1 * centroid

    store = DocumentStore()
    store.add_documents(texts, finetuned_embeddings)

    # 쿼리 임베딩도 같은 방식으로 조정
    def finetuned_embed(query_texts: list[str]) -> np.ndarray:
        """파인튜닝된 임베딩을 시뮬레이션한다."""
        return embed(query_texts)  # 쿼리는 조정 없이 base 사용

    metrics = _evaluate_retrieval(store, finetuned_embed, docs, doc_id_map)

    print(f"Precision@3: {metrics['precision_at_3']:.3f}")
    print(f"Recall@3:    {metrics['recall_at_3']:.3f}")
    print(f"MRR:         {metrics['mrr']:.3f}")

    return metrics


# ============================================================
# 3단계: LLM 쿼리 확장 시뮬레이션
# ============================================================


def step3_query_expansion() -> dict[str, float]:
    """쿼리 확장의 효과를 시뮬레이션하여 성능을 비교한다.

    Returns:
        평가 메트릭 딕셔너리.
    """
    print("\n" + "=" * 60)
    print("3단계: LLM 쿼리 확장 (시뮬레이션)")
    print("=" * 60)
    print("\n쿼리 확장이란?")
    print("- 원본 쿼리를 LLM으로 여러 변형을 생성")
    print("- 변형된 쿼리들로 검색하여 결과를 합산")
    print("- 표현이 다르지만 같은 의미의 문서를 더 잘 찾음\n")

    texts, docs = load_documents()
    doc_id_map = {d["id"]: i for i, d in enumerate(docs)}

    settings = get_settings()
    embed = create_embedding_fn(settings.embedding_model_name)

    doc_embeddings = embed(format_passages(texts))

    store = DocumentStore()
    store.add_documents(texts, doc_embeddings)

    # 시뮬레이션: 쿼리마다 수동 확장 버전 제공
    # (실제로는 LLM이 생성. experiments/exp_003 참고)
    EXPANDED_QUERIES = {
        "임베딩이란 무엇인가?": [
            "벡터 표현과 임베딩의 개념",
            "텍스트를 벡터로 변환하는 방법",
        ],
        "RAG 시스템은 어떻게 동작하는가?": [
            "검색 증강 생성의 작동 원리",
            "문서 검색 후 LLM으로 답변 생성하는 과정",
        ],
        "모델 성능을 어떻게 모니터링하는가?": [
            "ML 시스템의 성능 추적 방법",
            "프로덕션 모델 관제와 메트릭 수집",
        ],
        "트랜스포머 구조를 설명해주세요": [
            "셀프 어텐션 메커니즘 기반 신경망",
            "BERT GPT의 기반이 되는 아키텍처",
        ],
        "검색 품질을 높이는 방법은?": [
            "쿼리 확장과 검색 재현율 향상",
            "벡터 검색의 정확도를 개선하는 기법",
        ],
    }

    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    n = len(EVAL_QUERIES)

    for eq in EVAL_QUERIES:
        query = eq["query"]
        relevant_ids = eq["relevant_ids"]
        relevant_indices = {
            doc_id_map[rid] for rid in relevant_ids if rid in doc_id_map
        }

        # 원본 + 확장 쿼리 모두 검색
        all_queries = [query] + EXPANDED_QUERIES.get(query, [])
        all_query_embs = embed(format_queries(all_queries))

        # 모든 쿼리의 검색 결과를 합산 (점수 평균)
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        all_sims = cos_sim(all_query_embs, doc_embeddings)
        avg_sims = all_sims.mean(axis=0)

        top_indices = np.argsort(avg_sims)[::-1][:3]

        hits = sum(1 for idx in top_indices if idx in relevant_indices)
        precision = hits / 3
        recall = hits / len(relevant_indices) if relevant_indices else 0

        # MRR
        rr = 0.0
        for rank, idx in enumerate(top_indices, 1):
            if idx in relevant_indices:
                rr = 1.0 / rank
                break

        total_precision += precision
        total_recall += recall
        total_mrr += rr

    metrics = {
        "precision_at_3": total_precision / n,
        "recall_at_3": total_recall / n,
        "mrr": total_mrr / n,
    }

    print(f"Precision@3: {metrics['precision_at_3']:.3f}")
    print(f"Recall@3:    {metrics['recall_at_3']:.3f}")
    print(f"MRR:         {metrics['mrr']:.3f}")

    return metrics


# ============================================================
# 4단계: 종합 비교
# ============================================================


def step4_comparison(
    base: dict[str, float],
    finetuned: dict[str, float],
    expanded: dict[str, float],
) -> None:
    """세 방식의 성능을 비교한다."""
    print("\n" + "=" * 60)
    print("4단계: 종합 비교")
    print("=" * 60)

    header = f"{'방식':<20} {'Precision@3':>12} {'Recall@3':>10} {'MRR':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, m in [
        ("Base", base),
        ("Fine-tuned", finetuned),
        ("Query Expansion", expanded),
    ]:
        print(
            f"{name:<20} {m['precision_at_3']:>12.3f} {m['recall_at_3']:>10.3f} {m['mrr']:>8.3f}"
        )

    print("""
분석:
- Base 모델: 범용적이지만 도메인 특화 성능이 제한적
- Fine-tuned: 도메인 데이터로 학습하면 같은 카테고리 문서를 더 잘 찾음
- Query Expansion: 모델 변경 없이 검색 재현율을 높일 수 있음

실무 권장:
1. 먼저 Base 모델로 베이스라인 측정
2. 도메인 데이터가 충분하면 파인튜닝 시도
3. 파인튜닝이 어려우면 쿼리 확장으로 보완
4. 최적의 조합: Fine-tuned + Query Expansion
""")


# ============================================================
# 평가 헬퍼
# ============================================================


def _evaluate_retrieval(
    store: DocumentStore,
    embed_fn,
    docs: list[dict],
    doc_id_map: dict[str, int],
) -> dict[str, float]:
    """검색 성능을 Precision@3, Recall@3, MRR로 평가한다.

    Args:
        store: 문서 저장소.
        embed_fn: 임베딩 함수.
        docs: 원본 문서 리스트.
        doc_id_map: 문서 ID → 인덱스 매핑.

    Returns:
        precision_at_3, recall_at_3, mrr을 포함하는 딕셔너리.
    """
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    n = len(EVAL_QUERIES)

    for eq in EVAL_QUERIES:
        query_embedding = embed_fn(format_queries([eq["query"]]))
        results = retrieve(store, query_embedding, k=3)

        # 검색된 문서 인덱스
        retrieved_texts = {r[0] for r in results}
        relevant_indices = {
            doc_id_map[rid] for rid in eq["relevant_ids"] if rid in doc_id_map
        }
        relevant_texts = {docs[idx]["text"] for idx in relevant_indices}

        hits = len(retrieved_texts & relevant_texts)
        precision = hits / 3
        recall = hits / len(relevant_indices) if relevant_indices else 0

        # MRR (Mean Reciprocal Rank)
        rr = 0.0
        for rank, (text, _score) in enumerate(results, 1):
            if text in relevant_texts:
                rr = 1.0 / rank
                break

        total_precision += precision
        total_recall += recall
        total_mrr += rr

        print(f"  쿼리: '{eq['query'][:30]}...' → P={precision:.2f} R={recall:.2f}")

    return {
        "precision_at_3": total_precision / n,
        "recall_at_3": total_recall / n,
        "mrr": total_mrr / n,
    }


# ============================================================
# 실행
# ============================================================


def main() -> None:
    """세 가지 임베딩 방식을 비교하고 결과를 출력한다."""
    base_metrics = step1_base_retrieval()
    finetuned_metrics = step2_finetuned_simulation()
    expanded_metrics = step3_query_expansion()
    step4_comparison(base_metrics, finetuned_metrics, expanded_metrics)

    print("=" * 60)
    print("실습 과제")
    print("=" * 60)
    print("""
1. EVAL_QUERIES에 새로운 쿼리-정답 쌍을 추가하고 결과를 비교하세요.
2. 파인튜닝 시뮬레이션의 이동 비율(0.1)을 바꾸면 어떻게 되나요?
3. 쿼리 확장에서 확장 쿼리 수를 늘리면 성능이 어떻게 변하나요?
4. experiments/exp_001~003의 실제 실험 코드와 비교해보세요.
""")


if __name__ == "__main__":
    main()
