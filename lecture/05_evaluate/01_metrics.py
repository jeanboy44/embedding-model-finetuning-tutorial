"""
실습 2-1 (5교시): 평가셋과 지표
===============================

학습 목표:
- 평가셋(test)이 법령 단위로 나뉘어, 학습 때 본 적 없는 법령으로만 이루어졌음을 확인한다
- Recall@k · MRR@10 · nDCG@10을 작은 예로 직접 계산해 본다
- 판정 세 가지(doc: 정답 문서 일치 / article: 같은 조의 조각이면 정답 / multi: 판정으로 인정한 다른 정답도 정답)의 차이를 본다
- 학습 전 e5-small의 test 성적(출발점)을 명령 하나로 재고, 어떤 말투에서 약한지 읽는다

사전 준비 (없으면 스크립트가 받는 명령을 알려 주고 끝난다):
    uv run python scripts/data_version.py pull v1     # 코퍼스 + train/dev/test 분할
    models/multilingual-e5-small                      # scripts/download_model_hf.py
    uv run python scripts/finetuned_drive.py download   # data/processed/index/multilingual-e5-small.sqlite
    data/labels/dev.jsonl                             # dev 복수 정답 판정 (강사 Drive)

실행:
    uv run python lecture/05_evaluate/01_metrics.py          # test 전체를 받은 인덱스로 평가 (약 1분)
    uv run python lecture/05_evaluate/01_metrics.py --run    # 질문 일부로 test와 dev(복수 정답)를 직접 평가

결과는 experiments/results/lecture/05_evaluate/ (기본) 또는 임시 폴더(--run)에 쓴다. 받은 산출물은 건드리지 않는다.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ragkit.data import load_corpus, load_labels, question_key, relevance_key
from ragkit.evaluation import first_rank, question_metrics
from ragkit.retrieval import default_index_path
from ragkit.training import load_splits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
SPLITS = DATA / "splits"
LABELS = DATA / "labels" / "dev.jsonl"
MODEL = "intfloat/multilingual-e5-small"
INDEX = default_index_path(
    "multilingual-e5-small"
)  # ragkit evaluate가 재사용하는 인덱스
OUT = ROOT / "experiments" / "results" / "lecture" / "05_evaluate"
RUN_SAMPLE = 300  # --run에서 쓸 질문 수 (분할마다)
SEED = 42

GET_DATA = "uv run python scripts/data_version.py pull v1"
FROM_DRIVE = "강사 Drive에서 받아 {path}에 둔다"


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def require(items: list[tuple[Path, str]]) -> None:
    """필요한 파일이 없으면 무엇이 없고 어떻게 받는지 알려 주고 끝낸다."""
    missing = [(path, how) for path, how in items if not path.exists()]
    if not missing:
        return
    print("필요한 산출물이 없습니다.")
    for path, how in missing:
        print(f"  - {rel(path)}\n      받기: {how.format(path=rel(path))}")
    print(
        "준비 상태는 uv run python lecture/03_setup/01_doctor.py 로 한 번에 볼 수 있습니다."
    )
    raise SystemExit(1)


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def r5(group: dict, judge: str = "doc") -> str:
    return f"{group[judge]['recall@5']:.3f}"


sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
parser = argparse.ArgumentParser(description="실습 2-1: 평가셋과 지표")
parser.add_argument(
    "--run", action="store_true", help=f"질문 {RUN_SAMPLE}개로 test·dev를 직접 평가한다"
)
args = parser.parse_args()

require(
    [
        (CORPUS, GET_DATA),
        (SPLITS / "test.jsonl", GET_DATA),
        (SPLITS / "split_meta.json", GET_DATA),
        (
            INDEX,
            "uv run python scripts/finetuned_drive.py download"
            + " (없으면 ragkit evaluate가 7~13분 걸려 새로 만든다)",
        ),
    ]
)


# ============================================================
# 1. 평가셋: 법령 단위 분할
# ============================================================
section("1. 평가셋: 법령 단위로 나눈 train / dev / test")
docs = load_corpus(CORPUS)
by_id = {d["id"]: d for d in docs}
splits = load_splits(SPLITS)
meta = json.loads((SPLITS / "split_meta.json").read_text(encoding="utf-8"))
for name in ("train", "dev", "test"):
    print(
        f"  {name:5s} 질문 {len(splits[name]):6,}개 · 법령 {len(meta[name]['laws']):2d}개"
    )
overlap = set(meta["train"]["laws"]) & set(meta["test"]["laws"])
print(
    f"  train과 test에 함께 든 법령: {len(overlap)}개 → test는 학습 때 본 적 없는 법령으로만 이루어진다"
)
print(f"  검색 대상: 코퍼스 전체 {len(docs):,}개 문서 (정답 법령 안에서만 찾지 않는다)")
types = Counter(q.get("query_type", "unknown") for q in splits["test"])
print("  test 질문 유형: " + " · ".join(f"{t} {n:,}" for t, n in sorted(types.items())))
example = next(q for q in splits["test"] if q.get("query_type") == "situation")
print(f"\n  예) [{example['query_type']}] {example['query']}")
print(f"      정답: {example['positive_id']}")
print("  규칙: 모델·설정은 dev로 고르고, test는 마지막 비교에 한 번만 쓴다")


# ============================================================
# 2. 지표를 손으로: 질문 5개의 정답 순위
# ============================================================
section("2. 지표를 작은 예로 계산하기")
ranks = [1, 3, None, 7, 12]  # 질문 5개에서 정답이 나온 순위 (None = 100위 밖)
ks = (1, 5, 10)
rows = [question_metrics(rank, ks) for rank in ranks]
print("질문  정답 순위  R@1  R@5  R@10  MRR@10  nDCG@10")
for i, (rank, m) in enumerate(zip(ranks, rows), 1):
    print(
        f"  q{i}  {rank or '밖':>8}  {m['recall@1']:.0f}    {m['recall@5']:.0f}    {m['recall@10']:.0f}"
        f"     {m['mrr@10']:.3f}   {m['ndcg@10']:.3f}"
    )
mean = {key: float(np.mean([m[key] for m in rows])) for key in rows[0]}
print(
    f"  평균        {mean['recall@1']:.2f} {mean['recall@5']:.2f} {mean['recall@10']:.2f}"
    f"   {mean['mrr@10']:.3f}   {mean['ndcg@10']:.3f}"
)
print(
    "  · R@k   = 정답이 k위 안에 든 질문의 비율. 주요 지표는 R@5 (RAG가 LLM에 넘기는 조문이 5개)"
)
print("  · MRR@10 = 1/순위의 평균 (1위 1, 3위 0.33, 10위 밖 0)")
print("  · nDCG@10 = 1/log2(순위+1)의 평균. 순위가 내려갈수록 천천히 깎는다")


# ============================================================
# 3. 판정: doc · article · multi
# ============================================================
section("3. 무엇을 정답으로 볼까: doc · article · multi")
fragment = next(d for d in docs if d.get("parent_id") and d["id"] != d["parent_id"])
siblings = [d["id"] for d in docs if d.get("parent_id") == fragment["parent_id"]]
print(
    f"긴 조문은 항·호 단위 조각으로 나뉘어 있다. 예) {fragment['parent_id']} → 조각 {len(siblings)}개"
)
ranked = [s for s in siblings if s != fragment["id"]][:1] + [
    "다른법_법률_제1조",
    fragment["id"],
]
article_keys = [relevance_key(by_id[i]) if i in by_id else i for i in ranked]
print(f"  검색 결과 순서: {ranked}")
print(f"  정답: {fragment['id']}")
print(
    f"  doc 판정     → {first_rank(ranked, fragment['id'])}위 (정답 문서와 정확히 같아야 한다)"
)
print(
    f"  article 판정 → {first_rank(article_keys, fragment['parent_id'])}위 (같은 조의 다른 조각도 정답)"
)

if LABELS.exists():
    labels = load_labels(LABELS)
    with_alt = {qid: lab for qid, lab in labels.items() if lab.get("alt_positive_ids")}
    print(
        f"\n복수 정답 판정 파일 {rel(LABELS)}: dev 질문 {len(labels):,}개를 판정, "
        f"다른 정답이 인정된 질문 {len(with_alt)}개 ({len(with_alt) / len(splits['dev']):.1%})"
    )
    dev_by_qid = {question_key(q): q for q in splits["dev"]}
    qid = next((q for q in with_alt if q in dev_by_qid), None)
    if qid:
        q = dev_by_qid[qid]
        print(
            f"  예) {q['query']}\n      정답 {q['positive_id']} + 인정한 다른 정답 {with_alt[qid]['alt_positive_ids']}"
        )
    print(
        "  multi 판정 = 정답 또는 인정한 다른 정답이 나오면 정답. test에는 판정 파일이 없어 multi = doc"
    )
else:
    print(
        f"\n(복수 정답 판정 파일 {rel(LABELS)}이 없어 multi 예시는 건너뛴다. {FROM_DRIVE.format(path=rel(LABELS))})"
    )


# ============================================================
# 4. 출발점: 학습 전 e5-small을 test로
# ============================================================
if args.run:
    section(f"4. --run: 질문 {RUN_SAMPLE}개로 직접 평가")
    tmp = Path(tempfile.mkdtemp(prefix="lecture-02-1-"))
    rng = random.Random(SEED)
    small = {
        name: rng.sample(splits[name], min(RUN_SAMPLE, len(splits[name])))
        for name in ("test", "dev")
    }
    for name, rows_ in small.items():
        write_jsonl(tmp / "splits" / f"{name}.jsonl", rows_)
    print(
        f"임시 분할 폴더: {tmp / 'splits'} (test·dev에서 {RUN_SAMPLE}개씩 무작위, seed {SEED})"
    )
    out = tmp / "e5_test.json"
    seconds = ragkit(
        "evaluate",
        MODEL,
        "--split",
        "test",
        "--splits",
        str(tmp / "splits"),
        "--out",
        str(out),
    )
    print(f"  ({seconds:.0f}초)")
    if LABELS.exists():
        dev_out = tmp / "e5_dev_multi.json"
        print("\n같은 모델을 dev로, 복수 정답 판정을 붙여서:")
        ragkit(
            "evaluate",
            MODEL,
            "--split",
            "dev",
            "--splits",
            str(tmp / "splits"),
            "--labels",
            str(LABELS),
            "--out",
            str(dev_out),
        )
        dev = json.loads(dev_out.read_text(encoding="utf-8"))
        print(
            f"  dev 표본 R@5: doc {r5(dev)} → multi {r5(dev, 'multi')} (다른 정답을 인정한 만큼 오른다)"
        )
    print(
        f"\n표본 {RUN_SAMPLE}개라 전체 test 값과 조금 다르다. 전체 값은 --run 없이 실행해서 본다."
    )
else:
    section("4. 출발점: 학습 전 e5-small, test 전체")
    out = OUT / "multilingual-e5-small_test.json"
    seconds = ragkit("evaluate", MODEL, "--split", "test", "--out", str(out))
    print(
        f"  ({seconds:.0f}초. 코퍼스 임베딩은 받은 인덱스를 재사용하고, 질문만 임베딩했다)"
    )

result = json.loads(out.read_text(encoding="utf-8"))

section("5. 결과 읽기")
print(
    f"질문 {result['n']:,}개 · 모델 {result['model']} · 백엔드 {result['backend']} · {result['dim']}차원"
)
print(
    f"  doc     R@1 {result['doc']['recall@1']:.3f}  R@5 {r5(result)}  R@10 {result['doc']['recall@10']:.3f}"
    f"  MRR@10 {result['doc']['mrr@10']:.3f}  nDCG@10 {result['doc']['ndcg@10']:.3f}"
)
print(f"  article R@5 {r5(result, 'article')} (같은 조의 조각을 정답으로 쳐 주면)")
print(
    f"→ 질문 {result['n']:,}개 중 {result['doc']['recall@5']:.0%}만 정답을 5위 안에 올린다"
)

print("\n질문 유형별 R@5 (doc)")
for name, group in result["by_query_type"].items():
    print(f"  {name:10s} {r5(group)}  (n={group['n']:,})")
print(
    "  키워드는 조문과 단어가 겹쳐 쉽고, 상황 설명형(situation)에서 작은 모델이 무너진다"
)

print("\n테마별 R@5 (doc)")
for name, group in sorted(
    result["by_theme"].items(), key=lambda kv: kv[1]["doc"]["recall@5"]
):
    print(f"  {name:10s} {r5(group)}  (n={group['n']:,})")

types_by_qid = {question_key(q): q.get("query_type") for q in splits["test"]}
misses = [
    row
    for row in result["per_question"]
    if types_by_qid.get(row["qid"]) == "situation" and (row["doc_rank"] or 999) > 5
]
print(f"\n틀린 situation 질문 {len(misses):,}개 중 3개 (1위로 나온 문서와 정답 순위)")
for row in random.Random(SEED).sample(misses, min(3, len(misses))):
    top1 = (
        by_id.get(row["top10"][0], {}).get("title", row["top10"][0])
        if row["top10"]
        else "-"
    )
    print(f"  · {row['query']}")
    rank_text = f"{row['doc_rank']}위" if row["doc_rank"] else "100위 밖"
    print(f"      정답 {by_id[row['positive_id']]['title']} → {rank_text}")
    print(f"      1위  {top1}")
print(f"\n결과 JSON: {rel(out)} (질문별 순위·top-10은 per_question)")
print("다음: 02_query_expansion.py — 학습 없이 LLM으로 질문을 고쳐 주면 얼마나 오르나")
