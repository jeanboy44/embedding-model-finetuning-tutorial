"""검색 평가: 전체 코퍼스에서 질문마다 정답 문서의 순위를 구해 Recall@k, MRR@10, nDCG@10을 낸다.

임베딩 함수(embed_fn)만 받으므로 onnx / torch 백엔드 어느 쪽이든 쓸 수 있다.
판정은 두 가지다.
- doc: 정답 문서 id와 정확히 일치
- article: 같은 조(relevance_key)의 조각이면 정답. 같은 조의 조각이 여러 개 나오면 첫 등장만 센다
- multi: 정답 또는 alt_positive_ids(판정으로 인정한 다른 정답) 중 하나가 나오면 정답.
  partial_ids(판정에서 부분답으로 본 문서)도 순위에서 뺀다
related_ids(정답을 부분적으로 담은 문서)는 모든 판정에서 순위에서 뺀다.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np

from ragkit.data import doc_text, question_key, relevance_key

CUTOFF = 10  # MRR, nDCG를 계산하는 순위 한도
CANDIDATES = 100  # 질문마다 정렬할 상위 문서 수 (article 판정의 중복 제거 여유 포함)


def first_rank(ranked_keys: Sequence[str], positive_key: str) -> int | None:
    """중복 키는 첫 등장만 세어 정답 키의 순위(1부터)를 구한다. 없으면 None."""
    seen: set[str] = set()
    for key in ranked_keys:
        if key in seen:
            continue
        seen.add(key)
        if key == positive_key:
            return len(seen)
    return None


def question_metrics(rank: int | None, ks: Sequence[int]) -> dict[str, float]:
    """정답이 하나인 질문의 지표. Recall@k는 정답이 k위 안에 있으면 1."""
    metrics = {f"recall@{k}": float(rank is not None and rank <= k) for k in ks}
    if rank is not None and rank <= CUTOFF:
        metrics[f"mrr@{CUTOFF}"] = 1.0 / rank
        metrics[f"ndcg@{CUTOFF}"] = 1.0 / math.log2(rank + 1)
    else:
        metrics[f"mrr@{CUTOFF}"] = metrics[f"ndcg@{CUTOFF}"] = 0.0
    return metrics


def evaluate_retrieval(
    embed_fn: Callable[[list[str]], np.ndarray],
    corpus: list[dict],
    questions: list[dict],
    *,
    ks: Sequence[int] = (1, 5, 10),
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
    corpus_embeddings: np.ndarray | None = None,
    format_query: Callable[[str], str] | None = None,
    format_doc: Callable[[dict], str] | None = None,
) -> dict:
    """전체 코퍼스 대상 검색 성능을 잰다.

    Args:
        embed_fn: 텍스트 목록 → L2 정규화된 (N, dim) 배열.
        corpus: 문서 목록 (law_docs.json).
        questions: query, positive_id(, related_ids, query_type)를 가진 질문 목록.
        ks: Recall@k의 k 값들.
        query_prefix: 질문 앞 문구.
        passage_prefix: 문서 앞 문구.
        corpus_embeddings: 미리 계산한 코퍼스 임베딩 (corpus와 같은 순서). None이면 계산한다.
        format_query: 질문 → 임베딩 입력 문자열 (모델 프로필의 format_query). 주면 query_prefix 대신 쓴다.
        format_doc: 문서 → 임베딩 입력 문자열 (모델 프로필의 format_doc). 주면 passage_prefix 대신 쓴다.

    Returns:
        {"n", "doc": 지표, "article": 지표, "by_query_type": {유형: {"n", "doc", "article"}},
         "by_theme": {...}, "per_question": [{"query", "positive_id", "doc_rank", "article_rank"}]}

    Raises:
        ValueError: 질문이 없거나, positive_id가 코퍼스에 없는 질문이 있을 때.
    """
    by_id = _check(corpus, questions)
    format_query = format_query or (lambda text: query_prefix + text)
    format_doc = format_doc or (lambda doc: passage_prefix + doc_text(doc))
    if corpus_embeddings is None:
        corpus_embeddings = embed_fn([format_doc(doc) for doc in corpus])
    query_embeddings = embed_fn([format_query(q["query"]) for q in questions])
    scores = query_embeddings @ corpus_embeddings.T
    depth = min(CANDIDATES, len(corpus))
    top = np.argpartition(-scores, depth - 1, axis=1)[:, :depth]

    ids = [doc["id"] for doc in corpus]
    ranked = [
        [ids[i] for i in candidates[np.argsort(-row_scores[candidates], kind="stable")]]
        for candidates, row_scores in zip(top, scores)
    ]
    return _score(questions, by_id, ranked, ks)


def evaluate_index(
    index,
    embed_fn: Callable[[list[str]], np.ndarray],
    corpus: list[dict],
    questions: list[dict],
    *,
    ks: Sequence[int] = (1, 5, 10),
    query_prefix: str = "query: ",
    format_query: Callable[[str], str] | None = None,
) -> dict:
    """만들어 둔 벡터 인덱스(ragkit.retrieval.VectorIndex)로 검색 성능을 잰다.

    코퍼스를 다시 임베딩하지 않으므로 같은 모델을 여러 번 평가할 때 빠르고,
    1단계 RAG 실습과 같은 인덱스 파일을 쓴다. 결과 형식은 evaluate_retrieval과 같다.

    Args:
        index: .search(query_embedding, k) -> [SearchHit]를 가진 인덱스.
        embed_fn: 질문 임베딩 함수 (인덱스를 만든 모델과 같아야 한다).
        corpus: 인덱스를 만든 코퍼스. 정답 문서의 조·테마 조회와 검증에만 쓴다.
        questions: 질문 목록.
        ks: Recall@k의 k 값들.
        query_prefix: 질문 앞 문구.
        format_query: 질문 → 임베딩 입력 문자열 (모델 프로필). 주면 query_prefix 대신 쓴다.
            문서 형식은 인덱스를 만들 때(build_index의 format_doc) 이미 정해졌다.

    Returns:
        evaluate_retrieval과 같은 형식의 딕셔너리.
    """
    by_id = _check(corpus, questions)
    format_query = format_query or (lambda text: query_prefix + text)
    query_embeddings = embed_fn([format_query(q["query"]) for q in questions])
    depth = min(CANDIDATES, len(corpus))
    ranked = [[hit.id for hit in index.search(emb, k=depth)] for emb in query_embeddings]
    return _score(questions, by_id, ranked, ks)


def _check(corpus: list[dict], questions: list[dict]) -> dict[str, dict]:
    """질문이 있고 모든 positive_id가 코퍼스에 있는지 확인하고 id → 문서를 돌려준다."""
    if not questions:
        raise ValueError("평가할 질문이 없습니다.")
    by_id = {doc["id"]: doc for doc in corpus}
    missing = [q["positive_id"] for q in questions if q["positive_id"] not in by_id]
    if missing:
        raise ValueError(
            f"코퍼스에 없는 positive_id가 {len(missing)}개 있습니다 (예: {missing[0]}). "
            "ragkit.data.filter_questions로 먼저 거르세요."
        )
    return by_id


def _score(
    questions: list[dict],
    by_id: dict[str, dict],
    ranked_ids: list[list[str]],
    ks: Sequence[int],
) -> dict:
    """질문마다 순위가 매겨진 문서 id 목록으로 doc/article 지표를 계산하고 묶는다."""
    rows: list[dict] = []
    for question, order in zip(questions, ranked_ids):
        ignore = set(question.get("related_ids") or [])
        order = [doc_id for doc_id in order if doc_id not in ignore]
        positive = by_id[question["positive_id"]]
        doc_rank = first_rank(order, positive["id"])
        answers = {positive["id"], *(question.get("alt_positive_ids") or [])}
        partial = set(question.get("partial_ids") or [])
        multi_order = [doc_id for doc_id in order if doc_id not in partial]
        multi_rank = next((i + 1 for i, doc_id in enumerate(multi_order) if doc_id in answers), None)
        article_rank = first_rank(
            [relevance_key(by_id[doc_id]) if doc_id in by_id else doc_id for doc_id in order],
            relevance_key(positive),
        )
        rows.append(
            {
                "qid": question_key(question),
                "top10": order[:10],
                "query": question["query"],
                "positive_id": positive["id"],
                "query_type": question.get("query_type") or "unknown",
                "theme": positive.get("theme") or "unknown",
                "law": positive.get("category") or "unknown",
                "doc_rank": doc_rank,
                "article_rank": article_rank,
                "multi_rank": multi_rank,
                "doc": question_metrics(doc_rank, ks),
                "article": question_metrics(article_rank, ks),
                "multi": question_metrics(multi_rank, ks),
            }
        )

    result = _summarize(rows)
    result["by_query_type"] = _group(rows, "query_type")
    result["by_theme"] = _group(rows, "theme")
    result["by_law"] = _group(rows, "law")
    result["per_question"] = [
        {
            key: row[key]
            for key in ("qid", "query", "positive_id", "law", "doc_rank", "article_rank", "multi_rank", "top10")
        }
        for row in rows
    ]
    return result


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "doc": _mean([row["doc"] for row in rows]),
        "article": _mean([row["article"] for row in rows]),
        "multi": _mean([row["multi"] for row in rows]),
    }


def _group(rows: list[dict], field: str) -> dict[str, dict]:
    return {
        value: _summarize([row for row in rows if row[field] == value])
        for value in sorted({row[field] for row in rows})
    }


def paired_test(
    base: dict,
    other: dict,
    *,
    judge: str = "multi",
    k: int = 5,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """같은 질문 집합에 대한 두 평가 결과를 질문 단위로 짝지어 R@k 차이를 검정한다.

    Args:
        base: 기준 결과 (evaluate_retrieval/evaluate_index 결과, per_question에 qid가 있어야 한다).
        other: 비교 결과.
        judge: 순위 판정 (doc | article | multi).
        k: R@k의 k.
        n_boot: 대응 bootstrap 반복 수.
        seed: bootstrap seed.

    Returns:
        {"n", "base", "other", "delta", "ci95", "fixed", "broken", "mcnemar_p", "by_law": {법령: {n, base, other, delta}}}

    Raises:
        ValueError: 두 결과의 질문 집합(qid)이 다를 때.
    """
    from math import comb

    def hits(result: dict) -> dict[str, tuple[str, int]]:
        out = {}
        for row in result["per_question"]:
            rank = row.get(f"{judge}_rank")
            out[row["qid"]] = (row.get("law") or "unknown", int(rank is not None and rank <= k))
        return out

    a, b = hits(base), hits(other)
    if set(a) != set(b):
        raise ValueError(f"질문 집합이 다릅니다 (기준 {len(a)}개, 비교 {len(b)}개, 공통 {len(set(a) & set(b))}개)")
    qids = sorted(a)
    x = np.array([a[q][1] for q in qids])
    y = np.array([b[q][1] for q in qids])
    diff = y - x
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(qids), size=(n_boot, len(qids)))].mean(axis=1)
    fixed, broken = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    m = fixed + broken
    p = min(1.0, 2 * sum(comb(m, i) for i in range(min(fixed, broken) + 1)) / 2**m) if m else 1.0
    by_law = {}
    for law in sorted({a[q][0] for q in qids}):
        idx = [i for i, q in enumerate(qids) if a[q][0] == law]
        by_law[law] = {
            "n": len(idx),
            "base": float(x[idx].mean()),
            "other": float(y[idx].mean()),
            "delta": float(diff[idx].mean()),
        }
    return {
        "judge": judge,
        "k": k,
        "n": len(qids),
        "base": float(x.mean()),
        "other": float(y.mean()),
        "delta": float(diff.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "fixed": fixed,
        "broken": broken,
        "mcnemar_p": float(p),
        "by_law": by_law,
    }
