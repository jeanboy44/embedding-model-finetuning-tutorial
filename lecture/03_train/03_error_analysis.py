"""
실습 3-3 (6교시): 오답 분석, 무엇을 고쳤고 무엇이 남았나
========================================================

학습 목표:
- 학습 전 e5-small과 파인튜닝 모델(001 다시 캔 오답 3개)을 질문 단위로 짝지어, 고친 질문과 새로 틀린 질문을 본다
- 질문 유형 · 테마 · 법령 · 질문과 조문의 단어 겹침별로 어디가 얼마나 올랐는지 나눠 본다
- 여전히 틀리는 질문의 1위 오답이 정답과 어떤 관계인지(엉뚱한 법령의 비슷한 조문, 법률↔시행령 혼동 등) 분류한다
- 파인튜닝으로 안 고쳐지는 것(크게 놓친 질문, 말투 차이)을 확인하고 다음 가설을 한 문장으로 적는다
- 분류와 리포트는 AI에게 맡기고, DS는 가설과 판단에 집중하는 방식을 본다

분류(1위 오답과 정답의 관계, 순위 구간, 단어 겹침)는 ragkit.evaluation의 함수로 기계적으로 한다.

사전 준비:
    02_compare_runs.py를 먼저 실행해 실험 010 결과(질문별 순위)를 만든다
    (또는 강사가 나눠 준 experiments/exp_010_lecture_comparison/results/*.json)

실행:
    uv run python lecture/03_train/03_error_analysis.py          # 받은(또는 02에서 만든) test 전체 결과로 분석 (몇 초)
    uv run python lecture/03_train/03_error_analysis.py --run    # situation 질문 300개를 두 모델로 직접 평가해 분석 (1분 안쪽)

여전히 틀리는 질문 목록은 experiments/results/lecture/03_train/ 아래 JSONL로 남긴다(AI에게 분류를 맡길 때 쓴다).
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from ragkit.data import doc_text, load_corpus, load_questions, question_key
from ragkit.evaluation import (
    RANK_BUCKETS,
    RELATIONS,
    compare_hits,
    confusion_relation,
    query_overlap,
    rank_bucket,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
TEST = DATA / "splits" / "test.jsonl"
RECEIVED = ROOT / "experiments" / "exp_010_lecture_comparison" / "results"
LECTURE_OUT = ROOT / "experiments" / "results" / "lecture" / "03_train"
FROM_02 = LECTURE_OUT / "exp_010"
BASE, TUNED = (
    "multilingual-e5-small",
    "r001_A",
)  # 학습 전 / 실험 010에서 가장 좋은 파인튜닝 모델
LABELS = {BASE: "학습 전", TUNED: "001 파인튜닝"}
K = 5
RUN_SAMPLE = 300
SEED = 42

GET_DATA = "uv run python scripts/data_version.py pull v1"


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def pad(text: str, width: int) -> str:
    """한글(전각)을 두 칸으로 세어 왼쪽 정렬한다."""
    used = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - used, 1)


def ragkit(*args: str) -> float:
    """ragkit CLI를 저장소 루트에서 실행하고 걸린 시간(초)을 돌려준다."""
    shown = [rel(Path(a)) if a.startswith(str(ROOT)) else a for a in args]
    print(f"$ uv run ragkit {' '.join(shown)}")
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    start = time.perf_counter()
    done = subprocess.run(
        [str(Path(sys.executable).parent / "ragkit"), *args],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if done.returncode != 0:
        print(
            f"ragkit {args[0]}이 실패했습니다 (종료 코드 {done.returncode}). 위 메시지를 확인하세요."
        )
        raise SystemExit(done.returncode)
    return time.perf_counter() - start


def load_results(folder: Path) -> dict[str, dict]:
    """결과 폴더의 모델별 JSON을 모델 폴더 이름 → 결과로 읽는다."""
    found = {}
    for path in sorted(folder.glob("*.json")):
        if path.name == "comparison.json" or path.stem.endswith("_expand"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if "per_question" in result:
            found[Path(result["model"]).name] = result
    return found


def hit(rank: int | None) -> bool:
    return rank is not None and rank <= K


def title(doc_id: str) -> str:
    return docs[doc_id]["title"] if doc_id in docs else doc_id


sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
parser = argparse.ArgumentParser(description="실습 3-3: 오답 분석")
parser.add_argument(
    "--run",
    action="store_true",
    help=f"situation 질문 {RUN_SAMPLE}개를 두 모델로 직접 평가해 분석",
)
args = parser.parse_args()

for path in (CORPUS, TEST):
    if not path.exists():
        print(f"필요한 산출물이 없습니다: {rel(path)}\n  받기: {GET_DATA}")
        raise SystemExit(1)
docs = {d["id"]: d for d in load_corpus(CORPUS)}
questions = {question_key(q): q for q in load_questions(TEST)}


# ============================================================
# 0. 질문별 결과 불러오기
# ============================================================
section("0. 질문별 결과")
if args.run:
    tmp = Path(tempfile.mkdtemp(prefix="lecture-03-3-"))
    situation = [q for q in questions.values() if q.get("query_type") == "situation"]
    sample = random.Random(SEED).sample(situation, RUN_SAMPLE)
    subset = tmp / "situation_sample.jsonl"
    subset.write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in sample),
        encoding="utf-8",
    )
    config = tmp / "config.yaml"  # 두 모델만 비교하는 임시 실험 설정
    tuned_dir = ROOT / "models" / "finetuned" / TUNED
    if not tuned_dir.exists():
        print(f"모델이 없습니다: {rel(tuned_dir)} (강사 Drive에서 받는다)")
        raise SystemExit(1)
    config.write_text(
        yaml.safe_dump(
            {
                "name": "lecture_error_analysis",
                "questions": str(subset),
                "models": ["intfloat/multilingual-e5-small", str(tuned_dir)],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(f"situation 질문 {RUN_SAMPLE}개 (seed {SEED})를 학습 전 / {TUNED}로 평가한다")
    seconds = ragkit("compare", str(config), "--out-dir", str(tmp / "results"))
    print(f"  ({seconds:.0f}초)")
    results = load_results(tmp / "results")
    source = tmp / "results"
else:
    results, source = {}, None
    for folder in (RECEIVED, FROM_02):
        results = load_results(folder)
        if BASE in results and TUNED in results:
            source = folder
            break
    if source is None:
        print("실험 010의 질문별 결과가 없습니다. 먼저 실행:")
        print("  uv run python lecture/03_train/02_compare_runs.py")
        print(f"  (또는 강사가 나눠 준 결과 JSON을 {rel(RECEIVED)}/ 에 둔다)")
        raise SystemExit(1)
base, tuned = results[BASE], results[TUNED]
print(f"결과: {rel(source)} · 질문 {base['n']:,}개 · 판정 doc · R@{K}")
print(
    f"  {LABELS[BASE]} R@{K} {base['doc'][f'recall@{K}']:.3f} → {LABELS[TUNED]} {tuned['doc'][f'recall@{K}']:.3f}"
)


# ============================================================
# 1. 어디가 올랐나
# ============================================================
section("1. 어디가 올랐나: 유형 · 테마 · 법령")
for field, name in (("by_query_type", "질문 유형"), ("by_theme", "테마")):
    print(f"\n{name}별 R@{K}")
    rows = sorted(
        base[field],
        key=lambda g: (
            tuned[field][g]["doc"][f"recall@{K}"] - base[field][g]["doc"][f"recall@{K}"]
        ),
    )
    for g in rows:
        a, b = (
            base[field][g]["doc"][f"recall@{K}"],
            tuned[field][g]["doc"][f"recall@{K}"],
        )
        print(
            f"  {pad(g, 12)}{a:.3f} → {b:.3f}  ({b - a:+.3f}, n={base[field][g]['n']:,})"
        )

laws = sorted(
    [
        g for g in base["by_law"] if base["by_law"][g]["n"] >= 20
    ],  # 질문이 너무 적은 법령은 뺀다
    key=lambda g: (
        tuned["by_law"][g]["doc"][f"recall@{K}"]
        - base["by_law"][g]["doc"][f"recall@{K}"]
    ),
)
print(f"\n법령별 R@{K} (질문 20개 이상, 가장 적게 오른 3개 / 가장 많이 오른 3개)")
for g in dict.fromkeys(laws[:3] + laws[-3:]):
    a, b = (
        base["by_law"][g]["doc"][f"recall@{K}"],
        tuned["by_law"][g]["doc"][f"recall@{K}"],
    )
    print(
        f"  {pad(g, 42)}{a:.3f} → {b:.3f}  ({b - a:+.3f}, n={base['by_law'][g]['n']:,})"
    )

print("\n질문과 정답 조문의 단어 겹침(글자 2-gram)별 R@5")
overlap = {
    row["qid"]: query_overlap(row["query"], doc_text(docs[row["positive_id"]]))
    for row in base["per_question"]
}
edges = np.quantile(list(overlap.values()), [0.25, 0.5, 0.75])
tuned_rank = {row["qid"]: row["doc_rank"] for row in tuned["per_question"]}
tuned_by_quartile = []
for i, name in enumerate(("겹침 하위 25%", "25~50%", "50~75%", "상위 25%")):
    lo = -1.0 if i == 0 else edges[i - 1]
    hi = 2.0 if i == 3 else edges[i]
    group = [row for row in base["per_question"] if lo < overlap[row["qid"]] <= hi]
    a = np.mean([hit(row["doc_rank"]) for row in group])
    b = np.mean([hit(tuned_rank[row["qid"]]) for row in group])
    print(f"  {pad(name, 14)}{a:.3f} → {b:.3f}  ({b - a:+.3f}, n={len(group):,})")
    tuned_by_quartile.append(b)
same_order = tuned_by_quartile == sorted(tuned_by_quartile)
print(
    "  → 조문과 말이 겹치지 않는 일상 말투 질문이 가장 어렵다."
    + (" 파인튜닝 뒤에도 겹침이 적을수록 낮다" if same_order else "")
)


# ============================================================
# 2. 고친 질문 · 새로 틀린 질문
# ============================================================
groups = compare_hits(base, tuned, judge="doc", k=K)
section(
    f"2. 질문 단위로 짝짓기: 고침 {len(groups['fixed']):,} · 새로 틀림 {len(groups['broken']):,} · "
    f"둘 다 틀림 {len(groups['both_wrong']):,} · 둘 다 맞힘 {len(groups['both_right']):,}"
)
rng = random.Random(SEED)
for name, label in (("fixed", "고친 질문"), ("broken", "새로 틀린 질문")):
    print(f"\n{label} 예시")
    for row in rng.sample(groups[name], min(3, len(groups[name]))):
        before = f"{row['base_rank']}위" if row["base_rank"] else "100위 밖"
        after = f"{row['other_rank']}위" if row["other_rank"] else "100위 밖"
        print(
            f"  · {row['query']}\n      정답 {title(row['positive_id'])}: {before} → {after}"
        )
broken_ranks = Counter(rank_bucket(row["other_rank"]) for row in groups["broken"])
print(
    "\n새로 틀린 질문의 파인튜닝 후 정답 순위: "
    + " · ".join(f"{b} {broken_ranks[b]}" for b in RANK_BUCKETS if broken_ranks[b])
)
near = broken_ranks["6~10위"] / max(len(groups["broken"]), 1)
print(
    f"  → 새로 틀린 질문의 {near:.0%}는 6~10위로 살짝 밀린 것이다. "
    f"고친 질문은 새로 틀린 질문의 {len(groups['fixed']) / max(len(groups['broken']), 1):.1f}배"
)


# ============================================================
# 3. 여전히 틀리는 질문: 1위 오답은 무엇인가
# ============================================================
section(
    f"3. 여전히 틀리는 질문 ({LABELS[TUNED]} 기준 {sum(not hit(r) for r in tuned_rank.values()):,}개)"
)
failures = []
for row in tuned["per_question"]:
    if hit(row["doc_rank"]) or not row["top10"]:
        continue
    positive, wrong = docs[row["positive_id"]], docs.get(row["top10"][0])
    if wrong is None:
        continue
    failures.append(
        {
            "qid": row["qid"],
            "query": row["query"],
            "query_type": questions.get(row["qid"], {}).get("query_type", "unknown"),
            "positive_id": row["positive_id"],
            "positive_title": positive["title"],
            "rank": row["doc_rank"],
            "top1_id": wrong["id"],
            "top1_title": wrong["title"],
            "relation": confusion_relation(positive, wrong),
        }
    )
relations = Counter(f["relation"] for f in failures)
print("1위 오답과 정답의 관계")
for key, label in RELATIONS.items():
    print(
        f"  {pad(label, 34)}{relations[key]:5,}  ({relations[key] / len(failures):.0%})"
    )
other_law = relations["other_law_same_theme"] + relations["other_law_other_theme"]
print(
    f"  → 다른 법령 문서가 1위인 경우가 {other_law / len(failures):.0%}: 엉뚱한 법령의 비슷한 조문"
)
print(
    f"  → 같은 법의 법률↔시행령·시행규칙 혼동이 {relations['same_law_other_type'] / len(failures):.0%}"
)

buckets = Counter(rank_bucket(f["rank"]) for f in failures)
print("\n정답은 몇 위에 있었나")
for b in RANK_BUCKETS[1:]:
    print(f"  {pad(b, 10)}{buckets[b]:5,}  ({buckets[b] / len(failures):.0%})")
far = buckets["31~100위"] + buckets["100위 밖"]

print("\n파인튜닝 후에도 틀린 situation 질문 5개 (1위가 왜 헷갈렸을지 적어 본다)")
situation_fail = [f for f in failures if f["query_type"] == "situation"]
for f in rng.sample(situation_fail, min(5, len(situation_fail))):
    rank = f"{f['rank']}위" if f["rank"] else "100위 밖"
    print(f"  · {f['query']}")
    print(f"      정답 {f['positive_title']} ({rank})")
    print(f"      1위  {f['top1_title']}  ← {RELATIONS[f['relation']]}")

confused = next((f for f in failures if f["relation"] == "same_law_other_type"), None)
if confused:
    print("\n법률↔시행령 혼동 예")
    print(
        f"  · {confused['query']}\n      정답 {confused['positive_title']}\n      1위  {confused['top1_title']}"
    )
    print(
        "      법률은 '무엇을', 시행령은 '구체적으로 얼마나·어떻게'를 정한다. 질문이 어느 쪽을 묻는지 모델이 구분하지 못한다"
    )


# ============================================================
# 4. 파인튜닝의 한계
# ============================================================
section("4. 파인튜닝으로 안 고쳐지는 것")
print(
    f"  · 정답이 31위 밖인 실패 {far:,}개 ({far / len(failures):.0%}): 오답을 바꿔 가까운 경쟁 문서를 밀어내는 학습으로는 닿지 않는다"
)
print(
    f"    5위 안에 못 드는 질문이 모두 30위 안에서 올라와도 R@5 상한은 "
    f"{1 - far / base['n']:.2f}다 (질문 {base['n']:,}개 기준)"
)
print(
    "  · 질문과 조문의 말투 차이: 코퍼스 전체 조문에 일상 말투 질문을 만들어 학습하는 것(합성 질문)이 다음 가설이다"
)
print(
    "  · 추론(여러 조문을 이어야 하는 질문), 숫자 조건, 코퍼스에 없는 지식은 검색 모델 학습으로 고쳐지지 않는다"
)
print(
    "  · test 법령은 학습 때 보지 않았다. 본 법령에서 얻은 이득의 일부만 새 법령으로 옮겨 간다 (docs/PLAN.md error analysis)"
)


# ============================================================
# 5. AI에게 맡기고, DS는 판단한다
# ============================================================
section("5. 분류와 리포트는 AI에게, 가설과 판단은 DS가")
if not args.run:
    LECTURE_OUT.mkdir(parents=True, exist_ok=True)
out = (tmp if args.run else LECTURE_OUT) / f"failures_{TUNED}.jsonl"
out.write_text(
    "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in failures),
    encoding="utf-8",
)
print(
    f"여전히 틀리는 질문 {len(failures):,}개를 {rel(out)}에 남겼다 (질문 · 정답 · 1위 오답 · 관계)"
)
print("AI(Claude Code 등)에게 이렇게 맡긴다:")
print(
    f'  "{rel(out)}의 실패를 원인별로 분류해 줘. 정답 라벨이 틀린 경우, 1위 문서도 사실상 정답인 경우,'
)
print(
    '   질문이 모호한 경우, 진짜 검색 오류를 나누고, 유형별로 예시 3개와 비율을 표로 정리해 줘."'
)
print(
    "이 강의의 실험도 그렇게 했다: dev 오답 분석을 에이전트 4개(정량 분해 · 라벨 품질 · 검색 파이프라인 · 학습 전략)에 나눠 맡겼다."
)
print(
    "DS가 하는 일: 어떤 숫자를 믿을지(오차 범위, dev 편중), 어떤 제약을 둘지, 다음에 무엇을 할지를 정한다."
)
print(
    "\n내 가설 하나 (한 문장으로): 예) '법률↔시행령 쌍을 오답으로 주면 같은 법 혼동이 줄어든다'"
)
