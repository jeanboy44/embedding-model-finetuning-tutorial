"""
실습 1-3 (5교시): 평가셋과 지표
===============================

학습 목표:
- 평가셋(test)이 조 단위로 나뉘어, 학습 때 본 적 없는 조로만 이루어졌음을 확인한다 (법령은 54개 모두 학습에 있다)
- Recall@k · MRR@10 · nDCG@10을 작은 예로 직접 계산해 본다
- 판정 세 가지(doc: 정답 문서 일치 / article: 같은 조의 조각이면 정답 / multi: 판정으로 인정한 다른 정답도 정답)의 차이를 본다
- 학습 전 e5-small의 test 성적(출발점)을 명령 하나로 재고, 어떤 말투에서 약한지 읽는다

무엇을 하나 (기본 모드, 약 1분):
    1. 평가셋: 분할별 질문 · 조 · 법령 수, train ∩ test 조 0개, test 질문 유형 수와 situation 예 하나
    2. 지표를 손으로: 질문 5개의 정답 순위를 ragkit.evaluation.question_metrics에 넣어 R@k · MRR · nDCG를 계산
    3. 판정 세 가지: 항·호 조각 예로 doc · article 판정의 순위 차이, data/labels/dev.jsonl로 multi 판정
    4. 학습 전 e5 평가: `ragkit evaluate intfloat/multilingual-e5-small --split test` (받은 인덱스 재사용)
       (--run) test 300개 표본과 복수 정답 판정이 있는 dev 질문만 임시 폴더에 평가
    5. 결과 읽기: doc · article R@5, 질문 유형별 · 테마별 R@5, 틀린 situation 질문 예 3개

사전 준비 (없으면 스크립트가 받는 명령을 알려 주고 끝난다):
    uv run python scripts/data_version.py pull v1     # 코퍼스 + 질문
    uv run python lecture/05_data_eval/02_split.py    # 조 단위 분할 data/splits_article/ (없으면 만든다)
    models/multilingual-e5-small                      # scripts/download_model_hf.py
    uv run python scripts/finetuned_drive.py download   # data/processed/index/multilingual-e5-small.sqlite
    data/labels/dev.jsonl                             # dev 복수 정답 판정 (강사 Drive)

실행:
    uv run python lecture/05_data_eval/03_metrics.py          # test 전체를 받은 인덱스로 평가 (약 1분)
    uv run python lecture/05_data_eval/03_metrics.py --run    # 질문 일부로 test와 dev(복수 정답)를 직접 평가

결과는 experiments/results/lecture/05_data_eval/ (기본) 또는 임시 폴더(--run)에 쓴다. 받은 산출물은 건드리지 않는다.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cyclopts
import numpy as np
from cyclopts import Parameter
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ragkit.data import load_corpus, load_labels, question_key, relevance_key
from ragkit.evaluation import first_rank, question_metrics
from ragkit.retrieval import default_index_path
from ragkit.training import load_splits

sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
console = Console(emoji=False)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
SPLITS = DATA / "splits_article"  # 조 단위 분할 (02_split.py)
LABELS = DATA / "labels" / "dev.jsonl"
MODEL = "intfloat/multilingual-e5-small"
INDEX = default_index_path(
    "multilingual-e5-small"
)  # ragkit evaluate가 재사용하는 인덱스
OUT = ROOT / "experiments" / "results" / "lecture" / "05_data_eval"
RUN_SAMPLE = 300  # --run에서 쓸 질문 수 (분할마다)
SEED = 42

GET_DATA = "uv run python scripts/data_version.py pull v1"
GET_SPLITS = "uv run python lecture/05_data_eval/02_split.py"
FROM_DRIVE = "강사 Drive에서 받아 {path}에 둔다"


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold]{escape(title)}[/]", style="blue")


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
    lines = [
        f"  - {escape(rel(path))}\n      받기: [dim]{escape(how.format(path=rel(path)))}[/]"
        for path, how in missing
    ]
    lines.append(
        "준비 상태는 [dim]uv run python lecture/03_setup/01_doctor.py[/] 로 한 번에 볼 수 있습니다."
    )
    console.print(
        Panel("\n".join(lines), title="필요한 산출물이 없습니다", border_style="red")
    )
    raise SystemExit(1)


def ragkit(*args: str) -> float:
    """ragkit CLI를 저장소 루트에서 실행하고 걸린 시간(초)을 돌려준다."""
    shown = [rel(Path(a)) if a.startswith(str(ROOT)) else a for a in args]
    console.print(f"[dim]$ uv run ragkit {escape(' '.join(shown))}[/]")
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    start = time.perf_counter()
    done = subprocess.run(
        [str(Path(sys.executable).parent / "ragkit"), *args],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if done.returncode != 0:
        console.print(
            f"[red]ragkit {escape(args[0])}이 실패했습니다 (종료 코드 {done.returncode}).[/] 위 메시지를 확인하세요."
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


@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(
        False, description=f"질문 {RUN_SAMPLE}개로 test·dev를 직접 평가한다"
    )


def cli(*, options: Options | None = None) -> Options:
    """실습 1-3: 평가셋과 지표"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)

require(
    [
        (CORPUS, GET_DATA),
        (SPLITS / "test.jsonl", GET_SPLITS),
        (SPLITS / "split_meta.json", GET_SPLITS),
        (
            INDEX,
            "uv run python scripts/finetuned_drive.py download"
            + " (없으면 ragkit evaluate가 7~13분 걸려 새로 만든다)",
        ),
    ]
)


# ============================================================
# 1. 평가셋: 조 단위 분할
# ============================================================
# 02_split.py가 만든 data/splits_article/을 다시 읽어 분할별 질문 · 조 · 법령 수를 찍고,
# train과 test의 정답 조(ragkit.data.relevance_key)가 겹치지 않음을 다시 확인한다.
# 출력에서 볼 것: test 1,863개 = keyword 399 · question 701 · situation 763, 검색 대상은 코퍼스 전체.
section("1. 평가셋: 조 단위로 나눈 train / dev / test")
docs = load_corpus(CORPUS)
by_id = {d["id"]: d for d in docs}
splits = load_splits(SPLITS)
meta = json.loads((SPLITS / "split_meta.json").read_text(encoding="utf-8"))
for name in ("train", "dev", "test"):
    console.print(
        f"  {name:5s} 질문 {len(splits[name]):6,}개 · 조 {meta[name]['articles']:5,}개 · 법령 {len(meta[name]['laws']):2d}개",
        highlight=False,
    )


def articles(name: str) -> set[str]:
    return {relevance_key(by_id[q["positive_id"]]) for q in splits[name]}


overlap = articles("train") & articles("test")  # 0개여야 한다 (02_split.py)
console.print(
    f"  train과 test에 함께 든 조: [bold]{len(overlap)}[/]개 → test는 학습 때 본 적 없는 조로만 이루어진다 (법령은 모두 학습에 있다)",
    highlight=False,
)
console.print(
    f"  검색 대상: 코퍼스 전체 [bold]{len(docs):,}[/]개 문서 (정답 법령 안에서만 찾지 않는다)",
    highlight=False,
)
types = Counter(q.get("query_type", "unknown") for q in splits["test"])
console.print(
    "  test 질문 유형: "
    + " · ".join(f"{escape(t)} {n:,}" for t, n in sorted(types.items()))
)
# 가장 어려운 유형(situation)의 첫 질문
example = next(q for q in splits["test"] if q.get("query_type") == "situation")
console.print(
    f"\n  예) {escape('[' + example['query_type'] + ']')} {escape(example['query'])}"
)
console.print(f"      정답: {escape(example['positive_id'])}")
console.print("  규칙: 모델·설정은 dev로 고르고, test는 마지막 비교에 한 번만 쓴다")


# ============================================================
# 2. 지표를 손으로: 질문 5개의 정답 순위
# ============================================================
# 정답 순위만 알면 지표가 나온다는 것을 보이려고, 순위 1 · 3 · 밖 · 7 · 12인 질문 5개를
# ragkit.evaluation.question_metrics에 넣어 R@1 · R@5 · R@10 · MRR@10 · nDCG@10을 찍고 평균을 낸다.
# 출력에서 볼 것: 평균 R@5 0.40 · MRR@10 0.295 · nDCG@10 0.367, 순위가 내려갈 때 지표마다 깎이는 정도.
section("2. 지표를 작은 예로 계산하기")
ranks = [1, 3, None, 7, 12]  # 질문 5개에서 정답이 나온 순위 (None = 100위 밖)
ks = (1, 5, 10)  # R@k의 k
rows = [question_metrics(rank, ks) for rank in ranks]  # 질문마다 지표 dict
metric_table = Table(box=box.SIMPLE_HEAD, show_edge=False)
metric_table.add_column("질문")
metric_table.add_column("정답 순위", justify="right")
for col in ("R@1", "R@5", "R@10", "MRR@10", "nDCG@10"):
    metric_table.add_column(col, justify="right")
for i, (rank, m) in enumerate(zip(ranks, rows), 1):
    metric_table.add_row(
        f"q{i}",
        str(rank or "밖"),
        f"{m['recall@1']:.0f}",
        f"{m['recall@5']:.0f}",
        f"{m['recall@10']:.0f}",
        f"{m['mrr@10']:.3f}",
        f"{m['ndcg@10']:.3f}",
    )
mean = {key: float(np.mean([m[key] for m in rows])) for key in rows[0]}
metric_table.add_section()
metric_table.add_row(
    "평균",
    "",
    f"[bold]{mean['recall@1']:.2f}[/]",
    f"[bold]{mean['recall@5']:.2f}[/]",
    f"[bold]{mean['recall@10']:.2f}[/]",
    f"[bold]{mean['mrr@10']:.3f}[/]",
    f"[bold]{mean['ndcg@10']:.3f}[/]",
)
console.print(metric_table)
console.print(
    "  · R@k   = 정답이 k위 안에 든 질문의 비율. 주요 지표는 R@5 (작업자가 화면 한 번에 보는 결과가 5개)"
)
console.print("  · MRR@10 = 1/순위의 평균 (1위 1, 3위 0.33, 10위 밖 0)")
console.print("  · nDCG@10 = 1/log2(순위+1)의 평균. 순위가 내려갈수록 천천히 깎는다")


# ============================================================
# 3. 판정: doc · article · multi
# ============================================================
# 항·호로 나뉜 첫 조문(전기사업법 제2조, 5조각)으로 검색 결과 순서를 가정해
# doc 판정(정답 문서 그대로)과 article 판정(같은 조의 조각이면 정답)의 순위를 ragkit.evaluation.first_rank로 비교한다.
# 이어서 복수 정답 판정 파일(dev 641개 판정, 다른 정답 인정 44개)로 multi 판정이 무엇인지 보인다.
section("3. 무엇을 정답으로 볼까: doc · article · multi")
# id ≠ parent_id 인 첫 문서 = 나뉜 조각. siblings = 같은 조의 조각 전부
fragment = next(d for d in docs if d.get("parent_id") and d["id"] != d["parent_id"])
siblings = [d["id"] for d in docs if d.get("parent_id") == fragment["parent_id"]]
console.print(
    f"긴 조문은 항·호 단위 조각으로 나뉘어 있다. 예) {escape(fragment['parent_id'])} → 조각 {len(siblings)}개"
)
ranked = [s for s in siblings if s != fragment["id"]][:1] + [
    "다른법_법률_제1조",
    fragment["id"],
]  # 가정한 검색 결과: 1위 같은 조의 다른 조각, 2위 엉뚱한 조문, 3위 정답 문서
article_keys = [relevance_key(by_id[i]) if i in by_id else i for i in ranked]
console.print(f"  검색 결과 순서: {escape(str(ranked))}", highlight=False)
console.print(f"  정답: {escape(fragment['id'])}")
console.print(
    f"  doc 판정     → [bold]{first_rank(ranked, fragment['id'])}위[/] (정답 문서와 정확히 같아야 한다)"
)
console.print(
    f"  article 판정 → [bold]{first_rank(article_keys, fragment['parent_id'])}위[/] (같은 조의 다른 조각도 정답)"
)

if LABELS.exists():
    labels = load_labels(LABELS)
    # 정답 외에 다른 조문도 맞다고 인정한 질문
    with_alt = {qid: lab for qid, lab in labels.items() if lab.get("alt_positive_ids")}
    console.print(
        f"\n복수 정답 판정 파일 {escape(rel(LABELS))}: 질문 {len(labels):,}개를 판정, "
        f"다른 정답이 인정된 질문 [bold]{len(with_alt)}[/]개 ({len(with_alt) / len(labels):.1%})",
        highlight=False,
    )
    console.print("  (판정은 옛 법령 단위 dev 질문으로 했다. 지금 dev와 겹치는 질문은 일부다)")
    pool_by_qid = {
        question_key(q): q for name in ("train", "dev", "test") for q in splits[name]
    }
    qid = next((q for q in with_alt if q in pool_by_qid), None)
    if qid:
        q = pool_by_qid[qid]
        console.print(
            f"  예) {q['query']}\n      정답 {q['positive_id']} + 인정한 다른 정답 {with_alt[qid]['alt_positive_ids']}",
            markup=False,
            highlight=False,
        )
    console.print(
        "  multi 판정 = 정답 또는 인정한 다른 정답이 나오면 정답. test에는 판정 파일이 없어 multi = doc"
    )
else:
    console.print(
        f"\n[yellow](복수 정답 판정 파일 {escape(rel(LABELS))}이 없어 multi 예시는 건너뛴다. {escape(FROM_DRIVE.format(path=rel(LABELS)))})[/]"
    )


# ============================================================
# 4. 출발점: 학습 전 e5-small을 test로
# ============================================================
# 기본 모드: `ragkit evaluate` CLI를 하위 프로세스로 불러 test 1,863개 전체를 잰다 (약 1분).
# 코퍼스 임베딩은 받은 인덱스(data/processed/index/multilingual-e5-small.sqlite)를 재사용하고 질문만 임베딩한다.
# --run: test 300개 표본과 복수 정답 판정이 있는 dev 질문(81개)을 임시 분할 폴더에 써서 직접 평가한다.
if args.run:
    section(f"4. --run: 질문 {RUN_SAMPLE}개로 직접 평가")
    tmp = Path(tempfile.mkdtemp(prefix="lecture-05-3-"))  # 결과와 임시 분할을 둘 곳
    rng = random.Random(SEED)  # 표본도 seed 42로 뽑아 누가 돌려도 같은 300개
    labeled = {question_key(q) for q in splits["dev"]} & (
        set(load_labels(LABELS)) if LABELS.exists() else set()
    )  # 지금 dev 질문 중 판정 파일에 있는 것
    small = {
        "test": rng.sample(splits["test"], min(RUN_SAMPLE, len(splits["test"]))),
        # dev는 복수 정답 판정이 있는 질문만 (판정 파일이 옛 dev로 만들어져 지금 dev와 일부만 겹친다)
        "dev": [q for q in splits["dev"] if question_key(q) in labeled],
    }
    for name, rows_ in small.items():
        write_jsonl(tmp / "splits" / f"{name}.jsonl", rows_)
    console.print(
        f"임시 분할 폴더: [dim]{escape(str(tmp / 'splits'))}[/] (test {len(small['test'])}개 무작위 seed {SEED}, "
        f"dev는 판정이 있는 {len(small['dev'])}개)"
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
    console.print(f"  [dim]({seconds:.0f}초)[/]")
    if small["dev"]:
        dev_out = tmp / "e5_dev_multi.json"
        console.print("\n같은 모델을 dev로, 복수 정답 판정을 붙여서:")
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
        console.print(
            f"  dev 표본 R@5: doc [bold]{r5(dev)}[/] → multi [bold green]{r5(dev, 'multi')}[/] (다른 정답을 인정한 만큼 오른다)"
        )
    console.print(
        f"\n표본 {RUN_SAMPLE}개라 전체 test 값과 조금 다르다. 전체 값은 --run 없이 실행해서 본다."
    )
else:
    section("4. 출발점: 학습 전 e5-small, test 전체")
    out = OUT / "multilingual-e5-small_test.json"  # 받은 결과 파일과 다른 폴더에 쓴다
    seconds = ragkit(
        "evaluate", MODEL, "--split", "test", "--splits", str(SPLITS), "--out", str(out)
    )
    console.print(
        f"  [dim]({seconds:.0f}초. 코퍼스 임베딩은 받은 인덱스를 재사용하고, 질문만 임베딩했다)[/]"
    )

result = json.loads(out.read_text(encoding="utf-8"))  # ragkit evaluate가 쓴 결과 JSON


# ============================================================
# 5. 결과 읽기
# ============================================================
# 결과 JSON에서 doc 판정의 R@1 · R@5 · R@10 · MRR@10 · nDCG@10과 article R@5를 읽고,
# by_query_type · by_theme으로 어떤 말투 · 테마에서 약한지 본다. 마지막에 5위 안에 못 든
# situation 질문 중 3개를 뽑아 1위 문서와 정답 순위를 보인다 (실험 001의 000과 같은 값:
# R@5 0.545, keyword 0.739 / question 0.573 / situation 0.417).
section("5. 결과 읽기")
console.print(
    f"질문 {result['n']:,}개 · 모델 {escape(str(result['model']))} · 백엔드 {escape(str(result['backend']))} · {result['dim']}차원"
)
console.print(
    f"  doc     R@1 {result['doc']['recall@1']:.3f}  R@5 [bold]{r5(result)}[/]  R@10 {result['doc']['recall@10']:.3f}"
    f"  MRR@10 {result['doc']['mrr@10']:.3f}  nDCG@10 {result['doc']['ndcg@10']:.3f}",
    highlight=False,
)
console.print(
    f"  article R@5 [bold]{r5(result, 'article')}[/] (같은 조의 조각을 정답으로 쳐 주면)",
    highlight=False,
)
console.print(
    f"→ 질문 {result['n']:,}개 중 [bold]{result['doc']['recall@5']:.0%}[/]만 정답을 5위 안에 올린다",
    highlight=False,
)


def r5_table(groups: list[tuple[str, dict]], label: str) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column(label)
    table.add_column("R@5", justify="right")
    table.add_column("n", justify="right")
    for name, group in groups:
        table.add_row(escape(name), r5(group), f"{group['n']:,}")
    return table


console.print("\n질문 유형별 R@5 (doc)")
console.print(r5_table(list(result["by_query_type"].items()), "질문 유형"))
console.print(
    "  키워드는 조문과 단어가 겹쳐 쉽고, 상황 설명형(situation)에서 작은 모델이 무너진다"
)

console.print("\n테마별 R@5 (doc)")
console.print(
    r5_table(
        sorted(result["by_theme"].items(), key=lambda kv: kv[1]["doc"]["recall@5"]),
        "테마",
    )
)

types_by_qid = {question_key(q): q.get("query_type") for q in splits["test"]}
misses = [
    row
    for row in result["per_question"]
    if types_by_qid.get(row["qid"]) == "situation" and (row["doc_rank"] or 999) > 5
]  # doc_rank가 None(100위 밖)이거나 5위 밖인 situation 질문
console.print(
    f"\n틀린 situation 질문 [bold]{len(misses):,}[/]개 중 3개 (1위로 나온 문서와 정답 순위)",
    highlight=False,
)
for row in random.Random(SEED).sample(misses, min(3, len(misses))):  # 항상 같은 3개
    top1 = (
        by_id.get(row["top10"][0], {}).get("title", row["top10"][0])
        if row["top10"]
        else "-"
    )
    console.print(f"  · {escape(row['query'])}", highlight=False)
    rank_text = f"{row['doc_rank']}위" if row["doc_rank"] else "100위 밖"
    console.print(
        f"      정답 {escape(by_id[row['positive_id']]['title'])} → [yellow]{rank_text}[/]",
        highlight=False,
    )
    console.print(f"      1위  {escape(top1)}", highlight=False)
if args.run:
    shutil.rmtree(tmp, ignore_errors=True)  # 표본 결과는 임시 폴더에만 두고 지운다
    console.print("\n[dim]임시 폴더는 지웠다. 기본 모드 결과는 바뀌지 않았다.[/]")
else:
    console.print(
        f"\n[dim]결과 JSON: {escape(rel(out))} (질문별 순위·top-10은 per_question)[/]"
    )
console.print(
    "다음: 04_baseline_train.py — 작은 모델을 한 번 학습하면 얼마나 오르나 (001: test R@5 0.545 → 0.744)"
)
