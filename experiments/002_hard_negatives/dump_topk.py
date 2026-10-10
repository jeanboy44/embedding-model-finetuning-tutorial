"""한 분할의 질문마다 상위 50개 검색 결과와 코퍼스·질문 임베딩을 저장한다 (error analysis, hard negative 채굴 재료).

사용: uv run python runs/001_hard_negatives/dump_topk.py <모델 이름|폴더> <dev|train> <출력 폴더>
먼저 인덱스가 있어야 한다 (ragkit evaluate가 data/processed/index/에 만든다).
출력:
  topk_<split>.jsonl     질문마다 query, query_type, positive_id, related_ids, hard_negative_ids,
                         rank(doc, related 제외, 100위 밖은 null), top50 [[id, score]]
  corpus_emb.npy         코퍼스 임베딩 (corpus_ids.json 순서, L2 정규화)
  query_emb_<split>.npy  질문 임베딩 (topk 파일 순서)
  corpus_ids.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import sqlite3
import sqlite_vec

from ragkit.data import filter_questions, load_corpus, load_questions
from ragkit.embeddings import create_embedding_fn, get_profile
from ragkit.retrieval import default_index_path, model_key

model, split, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

docs = load_corpus("data/processed/law_docs.json")
questions, _ = filter_questions(load_questions(f"data/splits/{split}.jsonl"), {d["id"]: d for d in docs})

checkpoint = Path(model) if Path(model).is_dir() else None
index_path = default_index_path(model_key(model, checkpoint))
conn = sqlite3.connect(str(index_path))
conn.enable_load_extension(True)
sqlite_vec.load(conn)
dim = int(dict(conn.execute("SELECT key, value FROM meta"))["dim"])
rows = conn.execute("SELECT d.id, v.embedding FROM docs d JOIN vec_docs v ON d.rowid = v.rowid ORDER BY d.rowid").fetchall()
ids = [r[0] for r in rows]
corpus_emb = np.stack([np.frombuffer(r[1], dtype=np.float32, count=dim) for r in rows])
print(f"{index_path.name}: {len(ids)} docs, dim {dim}")

profile = get_profile(model)
backend = "st" if "st" in profile.backends and profile.backends[0] == "st" else "torch"
embed = create_embedding_fn(model, checkpoint_path=checkpoint, backend=backend, device="cpu")
query_emb = np.asarray(embed([profile.format_query(q["query"]) for q in questions]), dtype=np.float32)

scores = query_emb @ corpus_emb.T
top = np.argsort(-scores, axis=1)[:, :100]
with open(out / f"topk_{split}.jsonl", "w", encoding="utf-8") as f:
    hits5 = 0
    for q, cand, row in zip(questions, top, scores):
        ranked = [ids[i] for i in cand]
        related = set(q.get("related_ids") or [])
        ranked_wo = [i for i in ranked if i not in related]  # ragkit.evaluation과 같게 related는 순위에서 뺀다
        rank = ranked_wo.index(q["positive_id"]) + 1 if q["positive_id"] in ranked_wo else None
        hits5 += rank is not None and rank <= 5
        f.write(json.dumps({
            "query": q["query"], "query_type": q.get("query_type"), "positive_id": q["positive_id"],
            "related_ids": q.get("related_ids") or [], "hard_negative_ids": q.get("hard_negative_ids") or [],
            "rank": rank, "top50": [[ids[i], round(float(row[i]), 4)] for i in cand[:50]],
        }, ensure_ascii=False) + "\n")
print(f"{model} {split}: R@5 {hits5 / len(questions):.3f} (n={len(questions)})")
np.save(out / "corpus_emb.npy", corpus_emb)
np.save(out / f"query_emb_{split}.npy", query_emb)
(out / "corpus_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
