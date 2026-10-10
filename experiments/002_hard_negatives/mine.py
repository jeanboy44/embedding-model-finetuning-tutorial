"""001 hard negative 채굴: exp_009 임베딩으로 train 질문마다 train 법령 문서 중 상위 30개 후보를 뽑는다.

사용: uv run python runs/001_hard_negatives/mine.py candidates   # 후보 manifest, A negative, B 판정 입력
      uv run python runs/001_hard_negatives/mine.py finalize-b   # 판정 결과로 B negative와 추가 정답 행
입력: artifacts/exp_009/{corpus_emb.npy, corpus_ids.json, query_emb_train.npy, topk_train.jsonl} (dump_topk.py)
출력 (artifacts/):
  mined_candidates.jsonl  질문마다 후보 30개: id, 풀 안 순위, 점수, 정답 대비 점수비, 관계
  negatives_A.jsonl       qid, negative_ids, reasons (top / fill / repeat)
  judge/b_input.jsonl     B 판정 대상: 상위 3개 중 점수비 0.95 이상 (질문 단위, 후보 순서는 섞음)
  negatives_B.jsonl       qid, negative_ids, reasons, alt_positive_ids (finalize-b)
"""

import collections
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

from ragkit.data import load_corpus, load_questions, question_key

RUN = Path(__file__).parent
ART = RUN / "artifacts"
EMB = ART / "exp_009"
K_NEG, WINDOW, POOL_TOP, MARGIN = 3, 10, 30, 0.95
by = {d["id"]: d for d in load_corpus("data/processed/law_docs.json")}
ref = re.compile(r"법\s*(제\d+조(?:의\d+)?)")


def relation(pos_id: str, other_id: str) -> str:
    p, o = by[pos_id], by[other_id]
    if (p.get("parent_id") or p["id"]) == (o.get("parent_id") or o["id"]):
        return "같은 조의 다른 조각"
    if p["category"] == o["category"]:
        if p["law_type"] != o["law_type"]:
            linked = (o["article_no"] in ref.findall(p["text"])) or (p["article_no"] in ref.findall(o["text"]))
            return "같은 법 다른 종류(위임 연결)" if linked else "같은 법 다른 종류(연결 없음)"
        return "같은 법 같은 종류 다른 조"
    return "다른 법(같은 테마)" if p["theme"] == o["theme"] else "다른 법(다른 테마)"


def pick(cands: list[dict], keep: list[str]) -> tuple[list[str], list[str]]:
    """keep(먼저 고른 것) 뒤를 점수비 0.95 미만 후보로 채운다: 1~10위 → 11~30위 → 반복."""
    chosen, reasons = list(keep), ["top"] * len(keep)
    for lo, hi, why in ((1, WINDOW, "top"), (WINDOW + 1, POOL_TOP, "fill")):
        for c in cands:
            if len(chosen) >= K_NEG:
                break
            if lo <= c["rank"] <= hi and c["ratio"] < MARGIN and c["id"] not in chosen:
                chosen.append(c["id"])
                reasons.append(why)
    base = list(chosen)
    while chosen and len(chosen) < K_NEG:
        chosen.append(base[len(chosen) % len(base)])
        reasons.append("repeat")
    return chosen, reasons


def candidates() -> None:
    ids = json.loads((EMB / "corpus_ids.json").read_text(encoding="utf-8"))
    corpus_emb = np.load(EMB / "corpus_emb.npy")
    query_emb = np.load(EMB / "query_emb_train.npy")
    rows = [json.loads(line) for line in open(EMB / "topk_train.jsonl", encoding="utf-8")]
    train_laws = {by[r["positive_id"]]["category"] for r in load_questions("data/splits/train.jsonl")}
    pool = np.array([by[i]["category"] in train_laws for i in ids])
    index = {i: n for n, i in enumerate(ids)}
    print(f"후보 풀: train 법령 {len(train_laws)}개의 문서 {pool.sum()}개 / 전체 {len(ids)}개")

    out_c = open(ART / "mined_candidates.jsonl", "w", encoding="utf-8")
    out_a = open(ART / "negatives_A.jsonl", "w", encoding="utf-8")
    (ART / "judge").mkdir(exist_ok=True)
    out_j = open(ART / "judge" / "b_input.jsonl", "w", encoding="utf-8")
    rng = random.Random(42)
    stats = collections.Counter()
    rel_a = collections.Counter()
    for r, q in zip(rows, query_emb):
        qid = question_key(r)
        s = corpus_emb @ q
        pos_score = float(s[index[r["positive_id"]]])
        s_pool = np.where(pool, s, -np.inf)
        s_pool[index[r["positive_id"]]] = -np.inf
        top = np.argsort(-s_pool)[:POOL_TOP]
        cands = [
            {"id": ids[i], "rank": k + 1, "score": round(float(s[i]), 4),
             "ratio": round(float(s[i]) / pos_score, 4), "relation": relation(r["positive_id"], ids[i])}
            for k, i in enumerate(top)
        ]
        out_c.write(json.dumps({"qid": qid, "positive_id": r["positive_id"], "positive_score": round(pos_score, 4),
                                "candidates": cands}, ensure_ascii=False) + "\n")
        neg, why = pick(cands, [])
        stats.update(why)
        rel_a.update(next(c["relation"] for c in cands if c["id"] == n) for n in set(neg))
        out_a.write(json.dumps({"qid": qid, "negative_ids": neg, "reasons": why}, ensure_ascii=False) + "\n")
        judge = [c for c in cands[:K_NEG] if c["ratio"] >= MARGIN]
        stats["b_judge"] += len(judge)
        if judge:
            rng.shuffle(judge)
            out_j.write(json.dumps({
                "qid": qid, "query": r["query"], "positive": view(r["positive_id"], 700),
                "candidates": [view(c["id"], 600) for c in judge],
            }, ensure_ascii=False) + "\n")
    print("A 선택 사유:", dict(stats))
    print("A negative 관계 분포:", dict(rel_a.most_common()))


def view(doc_id: str, n: int) -> dict:
    d = by[doc_id]
    return {"id": doc_id, "title": d["title"], "text": d["text"][:n]}


def finalize_b() -> None:
    """판정 결과(judge/out/*.jsonl: qid, cand_id, label)로 B negative를 만든다.

    상위 3개에서 no와 0.95 미만은 그대로 두고, full·partial은 빼고 0.95 미만 다음 순위로 채운다.
    full 문서는 alt_positive_ids로 남겨 학습 행을 추가한다.
    """
    labels = {}
    for f in sorted((ART / "judge" / "out").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                labels[(x["qid"], x["cand_id"])] = x["label"]
    out = open(ART / "negatives_B.jsonl", "w", encoding="utf-8")
    stats, missing = collections.Counter(), 0
    for line in open(ART / "mined_candidates.jsonl", encoding="utf-8"):
        m = json.loads(line)
        keep, alt = [], []
        for c in m["candidates"][:K_NEG]:
            if c["ratio"] < MARGIN:
                keep.append(c["id"])
                continue
            lab = labels.get((m["qid"], c["id"]))
            if lab is None:
                missing += 1
                continue
            stats[lab] += 1
            if lab == "no":
                keep.append(c["id"])
            elif lab == "full":
                alt.append(c["id"])
        neg, why = pick(m["candidates"], keep)
        stats.update(why)
        out.write(json.dumps({"qid": m["qid"], "negative_ids": neg, "reasons": why, "alt_positive_ids": alt},
                             ensure_ascii=False) + "\n")
    if missing:
        sys.exit(f"판정이 없는 B 대상 쌍이 {missing}개 있습니다")
    print("B 판정·선택 사유:", dict(stats))


if __name__ == "__main__":
    {"candidates": candidates, "finalize-b": finalize_b}[sys.argv[1]]()
