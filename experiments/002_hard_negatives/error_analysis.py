"""001 0단계 에러 분석: exp_009(현재 최고)의 dev 오답 유형과 hard negative 채굴 규모.

사용: uv run python runs/001_hard_negatives/error_analysis.py
입력: artifacts/exp_009/topk_{dev,train}.jsonl (dump_topk.py), data/labels/dev.jsonl
출력: 표준 출력 (00_error_analysis.md에 옮긴다)
"""

import collections
import json
import sys
from pathlib import Path

from ragkit.data import load_labels, question_key

RUN = Path(__file__).parent
sys.path.insert(0, str(RUN))
ART = RUN / "artifacts" / "exp_009"
from common import by, pct, relation  # noqa: E402


def rows(split):
    return [json.loads(line) for line in open(ART / f"topk_{split}.jsonl", encoding="utf-8")]

labels = load_labels(Path("data/labels/dev.jsonl"))
dev = rows("dev")
miss = []
for r in dev:
    lab = labels.get(question_key(r), {})
    answers = {r["positive_id"], *lab.get("alt_positive_ids", [])}
    skip = set(r["related_ids"]) | set(lab.get("partial_ids", []))
    order = [i for i, _ in r["top50"] if i not in skip]
    hit = any(i in answers for i in order[:5])
    if not hit:
        miss.append((r, order))
print(f"dev 질문 {len(dev)}개, 복수 정답 기준 top-5 실패 {len(miss)}개 (R@5 {1 - len(miss) / len(dev):.3f})")

top1 = collections.Counter(relation(r["positive_id"], o[0]) for r, o in miss)
print("\n[1] 실패 질문의 1위 문서와 정답의 관계:\n ", pct(top1))
top5 = collections.Counter(relation(r["positive_id"], i) for r, o in miss for i in o[:5])
print("[2] 실패 질문의 top-5 오답 전체(5개씩)의 관계:\n ", pct(top5))

in_hn_top1 = sum(o[0] in set(r["hard_negative_ids"]) for r, o in miss)
in_hn_top5 = sum(any(i in set(r["hard_negative_ids"]) for i in o[:5]) for r, o in miss)
print(f"[3] 1위 오답이 기존 hard negative인 비율 {in_hn_top1 / len(miss):.1%}, top-5 안에 기존 HN이 하나라도 있는 비율 {in_hn_top5 / len(miss):.1%}")

ranks = collections.Counter(
    "6~10" if r["rank"] and r["rank"] <= 10 else "11~30" if r["rank"] and r["rank"] <= 30
    else "31~100" if r["rank"] else ">100" for r, _ in miss
)
print("[4] 실패 질문의 정답 순위(doc 기준):", pct(ranks))

# train: 채굴 규모. 상위 30개(정답 제외) 중 정답 점수의 95% 이상 = 가짜 negative 의심(판정 대상)
train = rows("train")
amb, amb_q, rel_amb = 0, 0, collections.Counter()
for r in train:
    scores = dict(r["top50"])
    pos = scores.get(r["positive_id"])
    cands = [(i, s) for i, s in r["top50"][:31] if i != r["positive_id"]][:30]
    if pos is None:  # 정답이 50위 밖: 기준 점수를 50위 점수로 둔다(보수적)
        pos = r["top50"][-1][1]
    flagged = [i for i, s in cands if s >= 0.95 * pos]
    amb += len(flagged)
    amb_q += bool(flagged)
    rel_amb.update(relation(r["positive_id"], i) for i in flagged)
print(f"\n[5] train {len(train)}문항, 후보 30개 중 정답 점수의 95% 이상(판정 대상): {amb}개 ({amb / len(train):.1f}/문항), 해당 질문 {amb_q}개")
print("    판정 대상의 관계:", pct(rel_amb))
hn_rank = collections.Counter()
for r in train:
    order = [i for i, _ in r["top50"]]
    for h in r["hard_negative_ids"]:
        k = order.index(h) + 1 if h in order else None
        hn_rank["1~5" if k and k <= 5 else "6~30" if k and k <= 30 else "31~50" if k else ">50"] += 1
print("[6] train 기존 hard negative의 현재 순위:", pct(hn_rank))
