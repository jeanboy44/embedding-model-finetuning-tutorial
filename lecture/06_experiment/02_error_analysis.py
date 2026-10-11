"""
실습 2-2 (6교시): 001 오답 분석, 다음 가설 찾기
===============================================

학습 목표:
- 학습 전 e5-small과 001을 질문 단위로 짝지어, 고친 질문과 새로 틀린 질문을 본다
- 질문 유형 · 테마 · 법령 · 질문과 조문의 단어 겹침별로 어디가 얼마나 올랐는지 나눠 본다
- 여전히 틀리는 질문의 1위 오답이 정답과 어떤 관계인지(엉뚱한 법령의 비슷한 조문, 법률↔시행령 혼동 등) 분류한다
- 학습 데이터에 우리가 준 오답(hard negative)이 모델에게 얼마나 쉬웠는지 보고, 다음 가설(실험 002)을 한 문장으로 적는다
- 분류와 리포트는 AI에게 맡기고, DS는 가설과 판단에 집중하는 방식을 본다

분류(1위 오답과 정답의 관계, 순위 구간, 단어 겹침)는 ragkit.evaluation의 함수로 기계적으로 한다.

사전 준비:
    01_compare_runs.py를 먼저 실행해 학습 전 / 001의 test 결과(질문별 순위)를 만든다
    (또는 강사가 나눠 준 experiments/archive/exp_010_lecture_comparison/results/*.json)

실행:
    uv run python lecture/06_experiment/02_error_analysis.py          # 01의 test 결과로 분석 (몇 초)
    uv run python lecture/06_experiment/02_error_analysis.py --run    # + train 질문 300개를 001로 검색해 우리가 준 오답의 순위를 직접 잰다 (1분 안쪽)

여전히 틀리는 질문 목록은 experiments/results/lecture/06_experiment/ 아래 JSONL로 남긴다(AI에게 분류를 맡길 때 쓴다).
"""

import json
import random
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import cyclopts
import numpy as np
from cyclopts import Parameter
from pydantic import BaseModel, Field

from ragkit.data import doc_text, load_corpus, load_questions, question_key
from ragkit.embeddings import create_embedding_fn
from ragkit.embeddings.profiles import get_profile
from ragkit.evaluation import (
    RANK_BUCKETS,
    RELATIONS,
    compare_hits,
    confusion_relation,
    query_overlap,
    rank_bucket,
)
from ragkit.retrieval import VectorIndex, default_index_path, model_key

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
TEST = DATA / "splits" / "test.jsonl"
TRAIN = DATA / "splits" / "train.jsonl"
RECEIVED = ROOT / "experiments" / "exp_010_lecture_comparison" / "results"
LECTURE_OUT = ROOT / "experiments" / "results" / "lecture" / "06_experiment"
FROM_01 = LECTURE_OUT / "exp_010"
TUNED_DIR = ROOT / "models" / "finetuned" / "001_baseline"
BASE, TUNED = "multilingual-e5-small", "001_baseline"  # 학습 전 / 기본 실험 001
LABELS = {BASE: "학습 전", TUNED: "001"}
K = 5
RUN_SAMPLE = 300
SEED = 42
DEPTH = 50  # 오답 순위를 볼 깊이 (이보다 아래면 "50위 밖")

GET_DATA = "uv run python scripts/data_version.py pull v1"
GET_002 = "uv run python scripts/finetuned_drive.py download 001_baseline"


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


def negative_bucket(rank: int | None) -> str:
    if rank is None:
        return f"{DEPTH}위 밖"
    return "1~5위" if rank <= 5 else "6~30위" if rank <= 30 else f"31~{DEPTH}위"


sys.stdout.reconfigure(line_buffering=True)
@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(
        False,
        description=f"train 질문 {RUN_SAMPLE}개를 001로 검색해 우리가 준 오답의 순위를 직접 잰다",
    )


def cli(*, options: Options | None = None) -> Options:
    """실습 2-2: 001 오답 분석"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)

for path, how in ((CORPUS, GET_DATA), (TEST, GET_DATA), (TRAIN, GET_DATA)):
    if not path.exists():
        print(f"필요한 산출물이 없습니다: {rel(path)}\n  받기: {how}")
        raise SystemExit(1)
docs = {d["id"]: d for d in load_corpus(CORPUS)}
questions = {question_key(q): q for q in load_questions(TEST)}


# ============================================================
# 0. 질문별 결과 불러오기
# ============================================================
section("0. 질문별 결과")
results, source = {}, None
for folder in (RECEIVED, FROM_01):
    results = load_results(folder)
    if BASE in results and TUNED in results:
        source = folder
        break
if source is None:
    print("학습 전 / 001의 질문별 결과가 없습니다. 먼저 실행:")
    print("  uv run python lecture/06_experiment/01_compare_runs.py")
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
# 5. 우리가 준 오답은 어땠나
# ============================================================
section("5. 우리가 준 오답(hard negative)은 모델이 헷갈리는 것이었나")
train = load_questions(TRAIN)
pairs = [(q, h) for q in train for h in q.get("hard_negative_ids", []) if h in docs]
same_law = sum(
    docs[h]["category"] == docs[q["positive_id"]]["category"] for q, h in pairs
)
print(
    f"train 질문 {len(train):,}개에 붙은 오답 {len(pairs):,}개 중 정답과 같은 법: {same_law / len(pairs):.0%}"
)
print("  질문을 만들 때 LLM이 같은 법에서만 골랐다. 그런데 위 3에서 1위 오답은")
print(
    f"  다른 법령 문서가 {other_law / len(failures):.0%}였다. 모델이 실제로 헷갈리는 것과 우리가 연습시킨 것이 다르다"
)

if args.run:
    if not TUNED_DIR.exists():
        print(f"모델이 없습니다: {rel(TUNED_DIR)}\n  받기: {GET_002}")
        raise SystemExit(1)
    index_path = default_index_path(model_key(str(TUNED_DIR), TUNED_DIR))
    if not index_path.exists():
        print(f"001 인덱스가 없습니다: {rel(index_path)}\n  받기: {GET_002}")
        raise SystemExit(1)
    sample = random.Random(SEED).sample(
        [q for q in train if q.get("hard_negative_ids")], RUN_SAMPLE
    )
    print(
        f"\n--run: train 질문 {RUN_SAMPLE}개 (seed {SEED})를 001로 검색해 오답의 순위를 잰다"
    )
    start = time.perf_counter()
    embed = create_embedding_fn(
        str(TUNED_DIR), backend="torch"
    )  # 받은 인덱스와 같은 백엔드
    profile = get_profile(TUNED_DIR)
    vectors = embed([profile.format_query(q["query"]) for q in sample])
    index = VectorIndex.open(index_path)
    ranks = Counter()
    for q, vec in zip(sample, vectors):
        order = [h.id for h in index.search(vec, k=DEPTH)]
        for neg in q["hard_negative_ids"]:
            ranks[negative_bucket(order.index(neg) + 1 if neg in order else None)] += 1
    index.close()
    total = sum(ranks.values())
    print(
        f"  ({time.perf_counter() - start:.0f}초) 오답 {total:,}개가 001 검색에서 몇 위인가"
    )
    for bucket in ("1~5위", "6~30위", f"31~{DEPTH}위", f"{DEPTH}위 밖"):
        print(f"    {pad(bucket, 10)}{ranks[bucket]:5,}  ({ranks[bucket] / total:.0%})")
    print(f"  → {DEPTH}위 밖 오답은 학습해도 배울 것이 거의 없다. 이미 쉬운 오답이다")
else:
    print(
        "\n같은 설정에 더 오래 학습한 003으로 잰 기록 (experiments/002_hard_negatives/00_error_analysis.md):"
    )
    print("  · train의 기존 오답 중 이미 50위 밖 40%, 6~30위 33%, 1~5위 18%")
    print("  · dev에서 틀린 질문의 1위 오답이 우리가 준 오답인 경우 4.7%")
    print("  001로 직접 재려면 --run")


# ============================================================
# 6. AI에게 맡기고, DS는 판단한다
# ============================================================
section("6. 분류와 리포트는 AI에게, 가설과 판단은 DS가")
LECTURE_OUT.mkdir(parents=True, exist_ok=True)
out = LECTURE_OUT / f"failures_{TUNED}.jsonl"
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
    "DS가 하는 일: 어떤 숫자를 믿을지(오차 범위, dev 편중), 어떤 제약을 둘지, 다음에 무엇을 할지를 정한다."
)
print("\n다음 가설 (한 문장으로): '모델이 헷갈리는 조문을 오답으로 주면 오른다'")
print(
    "  근거: 1위 오답의 절반이 다른 법, 정답의 다수가 30위 안. 우리가 준 오답은 같은 법이고 쉽다"
)
print("  이 가설을 에이전트가 돈 바퀴가 실험 002이다 → 03_hard_negatives.py")
