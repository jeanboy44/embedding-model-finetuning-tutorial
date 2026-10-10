"""조 단위 분할의 누수 검사: test가 train과 얼마나 닮았나, 닮은 질문을 빼도 001의 상승이 남나.

1. 문서: test 정답 문서마다 가장 비슷한 train 정답 문서 (법 ↔ 시행령이 같은 문장을 되풀이하는 경우)
2. 질문: test 질문마다 가장 비슷한 train 질문 (같은 LLM · 같은 가이드로 만든 질문끼리 닮는 경우)

유사도는 학습 전 e5 코사인(문서는 받은 인덱스의 임베딩)과 글자 3-gram 자카드(코사인 상위 10개 중 최대).
닮은 정도 구간별로 000(학습 전) · 001의 test R@5를 나눠 보고, 닮은 질문을 뺀 test로 다시 판정한다.

사용: uv run python experiments/001_baseline/leak_check.py
입력: data/splits_article, experiments/results/001_baseline/{000_base_test,001_test}.json
출력: experiments/results/001_baseline/leak_check.json
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

from ragkit.data import doc_text, load_corpus, question_key
from ragkit.embeddings import create_embedding_fn, get_profile
from ragkit.evaluation import paired_test
from ragkit.training.split import load_splits

MODEL = "intfloat/multilingual-e5-small"
INDEX = Path("data/processed/index/multilingual-e5-small.sqlite")
RESULTS = Path("experiments/results/001_baseline")
COS_CUTS = (0.95, 0.90)  # 코사인 구간 경계
JAC_CUT = 0.8  # 거의 같은 글
TOP = 10


def corpus_embeddings() -> dict[str, np.ndarray]:
    conn = sqlite3.connect(str(INDEX))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    dim = int(dict(conn.execute("SELECT key, value FROM meta"))["dim"])
    rows = conn.execute(
        "SELECT d.id, v.embedding FROM docs d JOIN vec_docs v ON d.rowid = v.rowid"
    ).fetchall()
    return {r[0]: np.frombuffer(r[1], dtype=np.float32, count=dim) for r in rows}


def grams(text: str) -> set[str]:
    text = "".join(text.split())
    return {text[i : i + 3] for i in range(len(text) - 2)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def nearest(a: np.ndarray, b: np.ndarray, a_texts: list[str], b_texts: list[str]) -> list[dict]:
    """a의 행마다 b에서 코사인 최대값과 그 이웃, 상위 TOP 이웃 중 3-gram 자카드 최대값."""
    scores = a @ b.T
    top = np.argsort(-scores, axis=1)[:, :TOP]
    b_grams = {}
    out = []
    for i, cand in enumerate(top):
        g = grams(a_texts[i])
        jac = 0.0
        for j in cand:
            if j not in b_grams:
                b_grams[j] = grams(b_texts[j])
            jac = max(jac, jaccard(g, b_grams[j]))
        out.append({"cos": float(scores[i, cand[0]]), "nn": int(cand[0]), "jac": jac})
    return out


def bucket(cos: float) -> str:
    if cos >= COS_CUTS[0]:
        return f">= {COS_CUTS[0]}"
    if cos >= COS_CUTS[1]:
        return f"{COS_CUTS[1]} ~ {COS_CUTS[0]}"
    return f"< {COS_CUTS[1]}"


def r5(rows: list[dict], qids: set[str]) -> float:
    hits = [r["doc_rank"] is not None and r["doc_rank"] <= 5 for r in rows if r["qid"] in qids]
    return sum(hits) / len(hits) if hits else float("nan")


def subset(result: dict, qids: set[str]) -> dict:
    return {**result, "per_question": [r for r in result["per_question"] if r["qid"] in qids]}


def main() -> None:
    docs = {d["id"]: d for d in load_corpus("data/processed/law_docs.json")}
    splits = load_splits(Path("data/splits_article"))
    train, test = splits["train"], splits["test"]
    base = json.loads((RESULTS / "000_base_test.json").read_text(encoding="utf-8"))
    tuned = json.loads((RESULTS / "001_test.json").read_text(encoding="utf-8"))

    # 1. 문서: test 정답 문서 ↔ train 정답 문서
    emb = corpus_embeddings()
    train_docs = sorted({q["positive_id"] for q in train})
    test_docs = sorted({q["positive_id"] for q in test})
    doc_near = nearest(
        np.stack([emb[i] for i in test_docs]),
        np.stack([emb[i] for i in train_docs]),
        [doc_text(docs[i]) for i in test_docs],
        [doc_text(docs[i]) for i in train_docs],
    )
    doc_info = {}
    for doc_id, n in zip(test_docs, doc_near):
        a, b = docs[doc_id], docs[train_docs[n["nn"]]]
        same_law = a["category"] == b["category"]
        doc_info[doc_id] = {
            **n,
            "nn": train_docs[n["nn"]],
            "relation": "같은 법령 · 법률↔시행령 등" if same_law and a["law_type"] != b["law_type"]
            else "같은 법령 · 같은 종류" if same_law
            else "다른 법령",
        }

    # 2. 질문: test 질문 ↔ train 질문 (학습 전 e5, query 앞 문구)
    profile = get_profile(MODEL)
    embed = create_embedding_fn(MODEL)
    q_emb = lambda rows: np.asarray(  # noqa: E731
        embed([profile.format_query(q["query"]) for q in rows]), dtype=np.float32
    )
    q_near = nearest(q_emb(test), q_emb(train), [q["query"] for q in test], [q["query"] for q in train])

    rows = []
    for q, n in zip(test, q_near):
        d = doc_info[q["positive_id"]]
        rows.append({
            "qid": question_key(q),
            "query": q["query"],
            "positive_id": q["positive_id"],
            "doc_cos": d["cos"], "doc_jac": d["jac"], "doc_nn": d["nn"], "doc_relation": d["relation"],
            "q_cos": n["cos"], "q_jac": n["jac"], "q_nn": train[n["nn"]]["query"],
        })

    report: dict = {"n": len(rows), "distribution": {}, "buckets": {}, "relation": {}}
    for key in ("doc_cos", "doc_jac", "q_cos", "q_jac"):
        v = np.array([r[key] for r in rows])
        report["distribution"][key] = {p: round(float(np.percentile(v, p)), 3) for p in (50, 90, 99)}
    report["distribution"]["doc_jac>=0.8"] = sum(r["doc_jac"] >= JAC_CUT for r in rows)
    report["distribution"]["q_jac>=0.8"] = sum(r["q_jac"] >= JAC_CUT for r in rows)

    for kind in ("doc_cos", "q_cos"):
        groups: dict[str, set[str]] = {}
        for r in rows:
            groups.setdefault(bucket(r[kind]), set()).add(r["qid"])
        report["buckets"][kind] = {
            name: {
                "n": len(ids),
                "000": round(r5(base["per_question"], ids), 3),
                "001": round(r5(tuned["per_question"], ids), 3),
            }
            for name, ids in sorted(groups.items())
        }
    for rel in sorted({r["doc_relation"] for r in rows}):
        ids = {r["qid"] for r in rows if r["doc_relation"] == rel}
        report["relation"][rel] = {
            "n": len(ids),
            "000": round(r5(base["per_question"], ids), 3),
            "001": round(r5(tuned["per_question"], ids), 3),
        }

    clean = {
        r["qid"] for r in rows
        if r["doc_cos"] < COS_CUTS[0] and r["q_cos"] < COS_CUTS[0]
        and r["doc_jac"] < JAC_CUT and r["q_jac"] < JAC_CUT
    }
    t = paired_test(subset(base, clean), subset(tuned, clean), judge="doc")
    report["clean"] = {k: t[k] for k in ("n", "base", "other", "delta", "ci95", "fixed", "broken", "mcnemar_p")}
    report["examples"] = {
        "doc": sorted(rows, key=lambda r: -r["doc_jac"])[:5],
        "q": sorted(rows, key=lambda r: -r["q_cos"])[:5],
    }

    out = RESULTS / "leak_check.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False, indent=2))
    for kind, ex in report["examples"].items():
        print(f"\n가장 닮은 예 ({kind})")
        for r in ex:
            pair = (r["positive_id"], r["doc_nn"]) if kind == "doc" else (r["query"], r["q_nn"])
            print(f"  cos {r[kind + '_cos']:.3f} jac {r[kind + '_jac']:.2f} | {pair[0]} ↔ {pair[1]}")
    print(f"\n결과 → {out}")


if __name__ == "__main__":
    main()
