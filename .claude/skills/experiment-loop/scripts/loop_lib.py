"""experiment-loop 스크립트가 같이 쓰는 함수: 경로, dev-a/b, 평가 결과 부분집합, 검정."""

import hashlib
import json
import random
from datetime import datetime, timezone
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SPLITS = ROOT / "data" / "splits_article"
CORPUS = ROOT / "data" / "processed" / "law_docs.json"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"
K = 5


def now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def qid(row: dict) -> str:
    """ragkit.data.question_key와 같은 식."""
    return hashlib.sha1((row["query"] + "\0" + row["positive_id"]).encode("utf-8")).hexdigest()[:12]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def make_dev_ab(seed: int = 42) -> tuple[Path, Path]:
    """dev를 정답 조(parent_id) 단위로 반씩 나눠 dev_a.jsonl · dev_b.jsonl을 만든다. 이미 있으면 그대로 둔다."""
    a_path, b_path = SPLITS / "dev_a.jsonl", SPLITS / "dev_b.jsonl"
    if a_path.exists() and b_path.exists():
        return a_path, b_path
    parent = {d["id"]: d["parent_id"] for d in json.loads(CORPUS.read_text(encoding="utf-8"))}
    dev = read_jsonl(SPLITS / "dev.jsonl")
    groups = sorted({parent[r["positive_id"]] for r in dev})
    random.Random(seed).shuffle(groups)
    a_groups = set(groups[: len(groups) // 2])
    write_jsonl(a_path, [r for r in dev if parent[r["positive_id"]] in a_groups])
    write_jsonl(b_path, [r for r in dev if parent[r["positive_id"]] not in a_groups])
    return a_path, b_path


def subset_qids(path: Path) -> set[str]:
    return {qid(r) for r in read_jsonl(path)}


def recall(rows: list[dict], judge: str = "doc", k: int = K) -> float:
    return sum(1 for r in rows if (r.get(f"{judge}_rank") or 10**9) <= k) / max(len(rows), 1)


def subset_eval(result: dict, qids: set[str]) -> dict:
    """ragkit evaluate 결과에서 질문 일부만 남기고 R@k를 다시 계산한다(paired-test에 그대로 쓸 수 있다)."""
    rows = [r for r in result["per_question"] if r["qid"] in qids]
    out = {k: v for k, v in result.items() if k not in ("per_question", "doc", "article", "multi", "by_query_type", "by_theme", "by_law")}
    out["n"] = len(rows)
    for judge in ("doc", "article", "multi"):
        out[judge] = {f"recall@{k}": recall(rows, judge, k) for k in (1, 5, 10)}
    out["per_question"] = rows
    return out


def mcnemar(fixed: int, broken: int) -> float:
    m = fixed + broken
    return min(1.0, 2 * sum(comb(m, i) for i in range(min(fixed, broken) + 1)) / 2**m) if m else 1.0


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[key] = running
    return out


def per_law_worse(base: dict, other: dict, judge: str = "doc", k: int = K) -> dict:
    """법령별 McNemar(Holm 보정). 유의하게 나빠진 법령 목록을 돌려준다."""
    a = {r["qid"]: r for r in base["per_question"]}
    b = {r["qid"]: r for r in other["per_question"]}
    laws: dict[str, list[int]] = {}
    for q in a.keys() & b.keys():
        law = a[q].get("law") or "unknown"
        x = int((a[q].get(f"{judge}_rank") or 10**9) <= k)
        y = int((b[q].get(f"{judge}_rank") or 10**9) <= k)
        f, br = laws.setdefault(law, [0, 0])
        laws[law] = [f + int(x == 0 and y == 1), br + int(x == 1 and y == 0)]
    raw = {law: mcnemar(f, br) for law, (f, br) in laws.items()}
    adj = holm(raw)
    worse = [law for law, (f, br) in laws.items() if br > f and adj[law] < 0.05]
    return {"worse": worse, "n_laws": len(laws), "holm_p": adj}


def append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
