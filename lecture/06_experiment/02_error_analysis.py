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

무엇을 하나:
    0. 01이 읽은(또는 직접 만든) 학습 전 / 001의 조 단위 test 결과 JSON을 읽는다 (질문별 순위 · top-10)
    1. 질문 유형 · 테마 · 법령 · 단어 겹침 4분위별 R@5를 학습 전 → 001로 비교한다
    2. ragkit.evaluation.compare_hits로 고침 · 새로 틀림 · 둘 다 틀림 · 둘 다 맞힘을 세고 예시를 보여 준다
    3. 001이 여전히 틀리는 질문의 1위 오답과 정답의 관계(confusion_relation) · 정답 순위 구간을 집계한다
    4. 파인튜닝으로 닿지 않는 범위(정답 31위 밖)와 R@5 상한을 계산한다
    5. train의 hard negative가 정답과 같은 법인 비율을 세고, 001로 잰 오답 순위 기록(실험 002 오답 분석)을 보여 준다
       --run이면 train 질문 300개를 001로 직접 검색해 오답의 순위를 잰다
    6. 실패 목록을 JSONL로 남기고, AI에게 맡길 분류 요청과 다음 가설을 적는다

사전 준비:
    01_compare_runs.py를 먼저 실행해 학습 전 / 001의 test 결과(질문별 순위)를 만든다
    (또는 강사가 나눠 준 experiments/results/001_baseline/{000_base_test,001_test}.json)

실행:
    uv run python lecture/06_experiment/02_error_analysis.py          # 01의 test 결과로 분석 (몇 초)
    uv run python lecture/06_experiment/02_error_analysis.py --run    # + train 질문 300개를 001로 검색해 우리가 준 오답의 순위를 직접 잰다 (1분 안쪽)

여전히 틀리는 질문 목록은 experiments/results/lecture/06_experiment/failures_001_baseline.jsonl로 남긴다(AI에게 분류를 맡길 때 쓴다).
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
SPLITS = DATA / "splits_article"  # 조 단위 분할 (5교시 02_split.py)
TEST = SPLITS / "test.jsonl"
TRAIN = SPLITS / "train.jsonl"
RECEIVED = (
    ROOT / "experiments" / "results" / "001_baseline"
)  # 강사가 나눠 준 실험 001 결과
LECTURE_OUT = ROOT / "experiments" / "results" / "lecture" / "06_experiment"
FROM_01 = LECTURE_OUT / "001_baseline"  # 01_compare_runs.py가 직접 평가했을 때 쓴 곳
TUNED_DIR = ROOT / "models" / "finetuned" / "001_baseline"
BASE, TUNED = "multilingual-e5-small", "001_baseline"  # 학습 전 / 기본 실험 001
LABELS = {BASE: "학습 전", TUNED: "001"}
K = 5
RUN_SAMPLE = 300  # --run에서 001로 검색할 train 질문 수
SEED = 42
DEPTH = 50  # 오답 순위를 볼 깊이 (이보다 아래면 "50위 밖")

GET_DATA = "uv run python scripts/data_version.py pull v1"
GET_SPLITS = "uv run python lecture/05_data_eval/02_split.py"
GET_001 = "uv run python scripts/finetuned_drive.py download 001_baseline"


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
    """결과 폴더의 모델별 JSON을 모델 폴더 이름 → 결과로 읽는다 (01_compare_runs.py와 같은 규칙)."""
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

for path, how in ((CORPUS, GET_DATA), (TEST, GET_SPLITS), (TRAIN, GET_SPLITS)):
    if not path.exists():
        print(f"필요한 산출물이 없습니다: {rel(path)}\n  받기: {how}")
        raise SystemExit(1)
docs = {d["id"]: d for d in load_corpus(CORPUS)}
questions = {
    question_key(q): q for q in load_questions(TEST)
}  # query_type을 붙이려고 test 질문 원본을 든다


# ============================================================
# 0. 질문별 결과 불러오기
# ============================================================
# 01이 읽은 순서와 같다: 받은 결과(RECEIVED) → 01이 직접 평가한 결과(FROM_01).
# 두 JSON의 per_question(질문별 doc_rank · top10)이 이 스크립트의 재료 전부다
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
print(
    f"결과: {rel(source)} · 조 단위 test 질문 {base['n']:,}개 · 법령 {len(base['by_law'])}개 · 판정 doc · R@{K}"
)
print(
    f"  {LABELS[BASE]} R@{K} {base['doc'][f'recall@{K}']:.3f} → {LABELS[TUNED]} {tuned['doc'][f'recall@{K}']:.3f}"
)


# ============================================================
# 1. 어디가 올랐나
# ============================================================
# 결과 JSON의 by_query_type · by_theme · by_law 집계를 그대로 비교하고(가장 적게 오른 순),
# 단어 겹침은 query_overlap(질문 글자 2-gram이 정답 조문에 나오는 비율)으로 4분위를 나눠 직접 센다
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
print(
    f"\n법령별 R@{K} (질문 20개 이상 {len(laws)}개 법령, 가장 적게 오른 3개 / 가장 많이 오른 3개)"
)
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
edges = np.quantile(list(overlap.values()), [0.25, 0.5, 0.75])  # 4분위 경계
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
# compare_hits가 같은 qid끼리 짝지어 네 묶음으로 나눈다. 01의 paired-test가 센 고침 · 망침과 같은 수다.
# 새로 틀린 질문의 정답 순위를 rank_bucket으로 묶어, 살짝 밀린 것(6~10위)이 얼마나 되는지 본다
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
# 001이 5위 안에 못 넣은 질문마다 top-10의 1위 문서를 꺼내 confusion_relation으로 정답과의 관계
# (같은 조 조각 · 같은 법 다른 조 · 법률↔시행령 · 다른 법 같은 테마 · 다른 법 다른 테마)를 매긴다.
# 이 목록(failures)이 6절에서 JSONL로 남는다
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
far = buckets["31~100위"] + buckets["100위 밖"]  # 오답 학습으로 닿기 어려운 범위

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
# 3절의 순위 구간에서 정답이 31위 밖인 실패 수(far)로, 가까운 경쟁 문서를 밀어내는 학습이
# 닿지 못하는 범위와 R@5 상한을 계산한다
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
    "  · test 조는 학습 때 보지 않았지만 54개 법령은 모두 학습에 있다. 이 점수는 '학습한 법령 안의 학습에 없던 조'에서의 성능이다"
)


# ============================================================
# 5. 우리가 준 오답은 어땠나
# ============================================================
# train 질문에 붙은 hard_negative_ids를 정답과 같은 법(category)인지 센다(질문을 만들 때 LLM이 골랐다).
# 기본 모드는 실험 002의 오답 분석(001로 train 표본 1,000개를 검색한 기록)을 보여 주고,
# --run이면 001 모델 + 받은 인덱스로 train 300개를 직접 검색해 각 오답의 순위를 잰다
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
        print(f"모델이 없습니다: {rel(TUNED_DIR)}\n  받기: {GET_001}")
        raise SystemExit(1)
    index_path = default_index_path(model_key(str(TUNED_DIR), TUNED_DIR))
    if not index_path.exists():
        print(f"001 인덱스가 없습니다: {rel(index_path)}\n  받기: {GET_001}")
        raise SystemExit(1)
    sample = random.Random(SEED).sample(
        [q for q in train if q.get("hard_negative_ids")], RUN_SAMPLE
    )  # 오답이 붙은 질문만 (조 단위 분할에서 오답이 모두 빠진 질문 437개는 제외)
    print(
        f"\n--run: train 질문 {RUN_SAMPLE}개 (seed {SEED})를 001로 검색해 오답의 순위를 잰다"
    )
    start = time.perf_counter()
    embed = create_embedding_fn(
        str(TUNED_DIR), backend="torch"
    )  # 받은 인덱스와 같은 백엔드
    profile = get_profile(TUNED_DIR)
    vectors = embed(
        [profile.format_query(q["query"]) for q in sample]
    )  # query: 접두어는 profile이 붙인다
    index = VectorIndex.open(index_path)
    ranks = Counter()
    for q, vec in zip(sample, vectors):
        order = [h.id for h in index.search(vec, k=DEPTH)]  # 상위 50개만 본다
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
        "\n001로 train 표본 1,000개를 검색해 잰 기록 (실험 002 오답 분석, experiments/002_hard_negatives/00_error_analysis.md):"
    )
    print(
        "  · train의 기존 오답 중 이미 50위 밖 39%, 31~50위 9%, 6~30위 31%, 1~5위 21%"
    )
    print(
        "  · 기존 오답의 관계: 같은 조의 다른 조각 50%, 같은 법 같은 종류 다른 조 48%, 법률↔시행령 3%, 다른 법 0%"
    )
    print(
        "  · dev에서 틀린 질문의 1위 오답이 우리가 준 오답인 경우 3.3%, 5위 안에 하나라도 있는 경우 17.1%"
    )
    print("  001로 직접 재려면 --run")


# ============================================================
# 6. AI에게 맡기고, DS는 판단한다
# ============================================================
# 3절의 실패 목록을 JSONL로 남긴다(질문 · 정답 · 1위 오답 · 관계). 분류는 AI에게 맡기고
# DS는 어떤 숫자를 믿을지와 다음 가설을 정한다. 이 가설이 실험 002다
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
    f"  근거: 1위 오답의 {other_law / len(failures):.0%}가 다른 법, 정답의 {1 - far / len(failures):.0%}가 30위 안. 우리가 준 오답은 같은 법이고 쉽다"
)
print("  이 가설을 에이전트가 돈 바퀴가 실험 002이다 → 03_hard_negatives.py")
