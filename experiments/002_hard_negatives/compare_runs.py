"""001 4단계 결과 분석: 두 run의 dev 실패 유형(1위 오답과 정답의 관계)과 고친/망친 질문을 비교한다.

사용: uv run python runs/001_hard_negatives/compare_runs.py <기준 run> <비교 run>   (예: r001_control_s42 r001_A)
입력: artifacts/<run>/topk_dev.jsonl (dump_topk.py), data/labels/dev.jsonl
출력: 표준 출력, artifacts/fixed_broken_<기준>_vs_<비교>.jsonl
"""

import collections
import json
import sys
from pathlib import Path

from ragkit.data import load_labels, question_key

sys.path.insert(0, str(Path(__file__).parent))
from common import pct, relation  # noqa: E402

RUN = Path(__file__).parent
ART = RUN / "artifacts"
base_name, other_name = sys.argv[1], sys.argv[2]
labels = load_labels(Path("data/labels/dev.jsonl"))


def judge(run: str) -> dict[str, dict]:
    out = {}
    for line in open(ART / run / "topk_dev.jsonl", encoding="utf-8"):
        r = json.loads(line)
        lab = labels.get(question_key(r), {})
        answers = {r["positive_id"], *lab.get("alt_positive_ids", [])}
        skip = set(r["related_ids"]) | set(lab.get("partial_ids", []))
        order = [i for i, _ in r["top50"] if i not in skip]
        rank = next((k + 1 for k, i in enumerate(order) if i in answers), None)
        out[question_key(r)] = {"row": r, "order": order, "rank": rank, "hit": rank is not None and rank <= 5}
    return out


a, b = judge(base_name), judge(other_name)
for name, res in ((base_name, a), (other_name, b)):
    miss = [v for v in res.values() if not v["hit"]]
    top1 = collections.Counter(relation(v["row"]["positive_id"], v["order"][0]) for v in miss)
    print(f"[{name}] 실패 {len(miss)}개, 1위 오답 관계: {pct(top1)}")

fixed = [k for k in a if not a[k]["hit"] and b[k]["hit"]]
broken = [k for k in a if a[k]["hit"] and not b[k]["hit"]]
print(f"\n고친 질문 {len(fixed)}개 / 망친 질문 {len(broken)}개")
for title, keys, res in (("고친 질문의 기준 run 1위 오답 관계", fixed, a), ("망친 질문의 비교 run 1위 오답 관계", broken, b)):
    c = collections.Counter(relation(res[k]["row"]["positive_id"], res[k]["order"][0]) for k in keys)
    print(f"  {title}: {pct(c)}")
qt = collections.Counter(a[k]["row"]["query_type"] for k in broken)
print("  망친 질문의 질문 유형:", dict(qt))
ranks = collections.Counter(
    "6~10" if b[k]["rank"] and b[k]["rank"] <= 10 else "11~30" if b[k]["rank"] and b[k]["rank"] <= 30 else "30+"
    for k in broken
)
print("  망친 질문의 새 정답 순위:", dict(ranks))
with open(ART / f"fixed_broken_{base_name}_vs_{other_name}.jsonl", "w", encoding="utf-8") as f:
    for kind, keys in (("fixed", fixed), ("broken", broken)):
        for k in keys:
            r = a[k]["row"]
            f.write(json.dumps({"kind": kind, "qid": k, "query": r["query"], "positive_id": r["positive_id"],
                                "rank_base": a[k]["rank"], "rank_other": b[k]["rank"],
                                "top1_base": a[k]["order"][0], "top1_other": b[k]["order"][0]}, ensure_ascii=False) + "\n")
