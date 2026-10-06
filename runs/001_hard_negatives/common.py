"""001 분석 스크립트 공용: 문서 조회, 정답과 다른 문서의 관계 6유형, 분포 출력."""

import collections
import re

from ragkit.data import load_corpus

by = {d["id"]: d for d in load_corpus("data/processed/law_docs.json")}
ref = re.compile(r"법\s*(제\d+조(?:의\d+)?)")


def relation(pos_id: str, other_id: str) -> str:
    """정답과 다른 문서의 관계. 위에서부터 먼저 맞는 것."""
    p, o = by[pos_id], by[other_id]
    if (p.get("parent_id") or p["id"]) == (o.get("parent_id") or o["id"]):
        return "같은 조의 다른 조각"
    if p["category"] == o["category"]:
        if p["law_type"] != o["law_type"]:
            linked = (o["article_no"] in ref.findall(p["text"])) or (p["article_no"] in ref.findall(o["text"]))
            return "같은 법 다른 종류(위임 연결)" if linked else "같은 법 다른 종류(연결 없음)"
        return "같은 법 같은 종류 다른 조"
    if p["theme"] == o["theme"]:
        return "다른 법(같은 테마)"
    return "다른 법(다른 테마)"


def pct(c: collections.Counter) -> str:
    n = sum(c.values())
    return ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in c.most_common())
