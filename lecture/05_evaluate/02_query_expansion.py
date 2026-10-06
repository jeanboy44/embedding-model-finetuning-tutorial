"""
실습 2-2 (5교시): 학습 없는 선택지, LLM 쿼리 확장
=================================================

학습 목표:
- LLM이 일상 말투 질문을 법령 용어 검색어로 바꿔 주는 쿼리 확장(실험 003)을 직접 본다
- 확장 전후로 정답 순위가 어떻게 바뀌는지 질문별로 비교한다
- 질문마다 LLM을 한 번 부르는 비용(토큰 · 지연 · 무료 한도)을 숫자로 확인한다
- 그래서 "학습해야 하는가"를 판단할 근거를 모은다

받은 확장 캐시(data/processed/query_expansion/)에 있는 질문만 쓰므로 기본 모드는 API 키 없이 돈다.
캐시는 아직 test의 일부뿐이다(Gemini 무료 등급이 하루 20회라 test 전체를 다 받지 못했다).

사전 준비 (없으면 스크립트가 받는 명령을 알려 주고 끝난다):
    uv run python scripts/data_version.py pull v1     # 코퍼스 + 분할
    data/processed/query_expansion/gemini-2.5-flash-lite.jsonl   # 확장 캐시 (강사 Drive)
    models/multilingual-e5-small                       # scripts/download_model_hf.py
    uv run python scripts/finetuned_drive.py download   # models/finetuned/r001_A + 그 인덱스
    (--run에서 새로 확장하려면) .env에 GEMINI_API_KEY

실행:
    uv run python lecture/05_evaluate/02_query_expansion.py          # 캐시에 있는 질문으로 비교 (1분 안쪽)
    uv run python lecture/05_evaluate/02_query_expansion.py --run    # 키가 있으면 질문 3개를 새로 확장해 함께 비교

받은 캐시는 고치지 않는다. --run의 새 확장은 임시 폴더의 캐시 사본에만 쌓는다.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

from ragkit.config import get_settings
from ragkit.data import load_corpus, load_questions
from ragkit.rag.query_expansion import (
    ExpansionUnavailable,
    default_cache_path,
    expand_queries,
    expansion_summary,
    load_cache,
    search_text,
)
from ragkit.retrieval import default_index_path, model_key

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CORPUS = DATA / "processed" / "law_docs.json"
TEST = DATA / "splits" / "test.jsonl"
CONFIG = ROOT / "experiments" / "exp_003_llm_query_expansion" / "config.yaml"
FINETUNED = (
    ROOT / "models" / "finetuned" / "r001_A"
)  # exp_003 설정의 세 번째 모델 (다음 교시 미리보기)
OUT = ROOT / "experiments" / "results" / "lecture" / "05_evaluate" / "query_expansion"
NEW_EXPANSIONS = 3  # --run에서 새로 부를 LLM 호출 수 (무료 등급 하루 20회)
FREE_PER_DAY = 20  # gemini-2.5-flash-lite 무료 등급 하루 호출 한도 (2026-09 확인)
FREE_RPM = 15  # 무료 등급 분당 호출 한도

GET_DATA = "uv run python scripts/data_version.py pull v1"
FROM_DRIVE = "강사 Drive에서 받아 {path}에 둔다"
PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here"}


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
        "준비 상태는 uv run python lecture/03_setup/01_check_env.py 로 한 번에 볼 수 있습니다."
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


def pad(text: str, width: int) -> str:
    """한글(전각)을 두 칸으로 세어 왼쪽 정렬한다."""
    used = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - used, 1)


def rank_text(rank: int | None) -> str:
    return f"{rank}위" if rank else "100위 밖"


def new_expansions(questions: list[dict], cache: Path) -> int:
    """캐시에 없는 test 질문 몇 개를 LLM으로 확장해 cache(임시 사본)에 쌓는다. 받은 수를 돌려준다."""
    from google.genai import errors

    key = get_settings().gemini_api_key
    if key in PLACEHOLDER_KEYS:
        print(
            "GEMINI_API_KEY가 없어 새 확장은 건너뛴다. .env에 키를 넣으면 --run이 질문 3개를 새로 확장한다."
        )
        return 0
    print(
        f"질문 {len(questions)}개를 새로 확장한다 (LLM 호출 {len(questions)}회, 분당 {FREE_RPM}회 간격)"
    )
    try:
        got = expand_queries(
            [q["query"] for q in questions], cache, min_interval_s=60 / FREE_RPM
        )
    except ExpansionUnavailable as e:
        print(f"확장할 수 없습니다: {e}")
        return 0
    except errors.APIError as e:
        hint = (
            "무료 한도(하루 20회)를 넘었을 수 있다. 내일 다시 하거나 캐시만으로 진행한다"
            if e.code == 429
            else ("키가 맞는지(.env의 GEMINI_API_KEY), 네트워크가 되는지 확인한다")
        )
        print(
            f"Gemini API 오류 ({e.code}): {e.message}\n  → {hint}. 지금까지 받은 확장은 {rel(cache)}에 남아 있다."
        )
        return 0
    for q in questions:
        item = got[q["query"]]
        print(
            f"  · {q['query']}\n      → {item.expansion}  ({item.input_tokens}+{item.output_tokens} 토큰, {item.latency_s:.1f}초)"
        )
    return len(questions)


sys.stdout.reconfigure(
    line_buffering=True
)  # ragkit 하위 프로세스 출력과 순서가 섞이지 않게
parser = argparse.ArgumentParser(description="실습 2-2: LLM 쿼리 확장")
parser.add_argument(
    "--run",
    action="store_true",
    help=f"키가 있으면 질문 {NEW_EXPANSIONS}개를 새로 확장해 함께 비교",
)
args = parser.parse_args()

settings = get_settings()
LLM = settings.gemini_model_name
CACHE = default_cache_path(LLM)
require(
    [
        (CORPUS, GET_DATA),
        (TEST, GET_DATA),
        (CACHE, FROM_DRIVE),
        (
            default_index_path("multilingual-e5-small"),
            "uv run python scripts/finetuned_drive.py download",
        ),
        (FINETUNED, "uv run python scripts/finetuned_drive.py download"),
    ]
)
finetuned_index = default_index_path(model_key(str(FINETUNED), FINETUNED))
if not finetuned_index.exists():
    print(f"참고: {rel(finetuned_index)}가 없어 r001_A 인덱스를 새로 만든다 (7~13분).")


# ============================================================
# 1. 받은 확장 캐시: 무엇이 들어 있나
# ============================================================
section("1. 쿼리 확장: LLM이 질문을 법령 말로 바꿔 준다")
test = load_questions(TEST)
cache = load_cache(CACHE, llm=LLM)
cached = [q for q in test if q["query"] in cache]
print(f"확장 캐시 {rel(CACHE)} ({LLM})")
print(
    f"  test 질문 {len(test):,}개 중 캐시에 있는 질문 {len(cached)}개 ({len(cached) / len(test):.1%})"
)
print(
    f"  무료 등급은 하루 {FREE_PER_DAY}회라 test 전체를 받으려면 약 {len(test) / FREE_PER_DAY:.0f}일이 걸린다."
)
print(
    "  → 그래서 test 전체에서 잰 '쿼리 확장' 행은 아직 비어 있다. 아래 비교는 이 몇 개 질문으로만 한다."
)
laws = sorted({q["positive_id"].split("_")[0] for q in cached})
print(f"  캐시 질문의 정답 법령: {', '.join(laws)} (한쪽에 몰려 있어 일반화할 수 없다)")

print("\n예시 (원래 질문 → 확장 검색어. 검색은 '원문 + 확장어'로 한다)")
for q in cached[:4]:
    item = cache[q["query"]]
    print(f"  · [{q.get('query_type')}] {q['query']}\n      → {item.expansion}")
print(
    f"\n실제로 검색에 쓰는 문장 예: {search_text(cached[0]['query'], cache[cached[0]['query']].expansion)!r}"
)


# ============================================================
# 2. 질문당 비용
# ============================================================
section("2. 질문당 비용: LLM 호출 1회")
summary = expansion_summary([cache[q["query"]] for q in cached])
latencies = sorted(cache[q["query"]].latency_s for q in cached)
print(f"  LLM 호출        질문당 {summary['llm_calls_per_query']}회")
print(f"  입력 토큰       평균 {summary['input_tokens_mean']:.0f} (프롬프트 + 질문)")
print(f"  출력 토큰       평균 {summary['output_tokens_mean']:.0f}")
print(
    f"  지연            평균 {summary['latency_s_mean']:.1f}초 · 가장 느린 것 {latencies[-1]:.1f}초 "
    f"(임베딩 검색은 질문당 수십 ms)"
)
print(
    f"  하루 1만 건이면 LLM 호출 1만 회, 입력 약 {summary['input_tokens_mean'] * 10_000 / 1e6:.1f}M 토큰이 매일 든다"
)
print("  → 학습이 필요 없는 대신, 사용자가 늘수록 비용과 지연이 함께 는다")


# ============================================================
# 3. 원문 vs 확장: 같은 질문으로 비교
# ============================================================
tmp = Path(tempfile.mkdtemp(prefix="lecture-02-2-"))
cache_for_run = CACHE
questions = cached
if args.run:
    section(f"3-0. --run: 캐시에 없는 질문 {NEW_EXPANSIONS}개를 새로 확장")
    cache_for_run = tmp / CACHE.name
    shutil.copy(CACHE, cache_for_run)  # 받은 캐시는 그대로 두고 사본에 쌓는다
    fresh = [
        q
        for q in test
        if q["query"] not in cache and q.get("query_type") == "situation"
    ][:NEW_EXPANSIONS]
    if new_expansions(fresh, cache_for_run):
        questions = cached + fresh

subset = tmp / "questions.jsonl"
subset.write_text(
    "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions),
    encoding="utf-8",
)
out_dir = tmp / "results" if args.run else OUT
section(f"3. 원문 vs 확장 검색 (질문 {len(questions)}개, 실험 003 설정)")
print(
    "실험 003은 e5 / e5 + 쿼리 확장 / 파인튜닝 모델(r001_A)을 같은 질문으로 비교한다."
)
print("질문을 캐시에 있는 것으로 바꿔(--questions) 키 없이 돌린다.")
seconds = ragkit(
    "compare",
    str(CONFIG),
    "--questions",
    str(subset),
    "--out-dir",
    str(out_dir),
    "--expansion-cache",
    str(cache_for_run),
)
print(f"  ({seconds:.0f}초)")

table = json.loads((out_dir / "comparison.json").read_text(encoding="utf-8"))
print("\n  " + pad("모델", 30) + "R@1    R@5    R@10   MRR@10  질문당 LLM 호출")
labels = {}
for row in table["models"]:
    if row.get("expansion"):
        label = "e5-small + 쿼리 확장"
    elif Path(row["model"]).is_dir():
        label = f"{Path(row['model']).name} (파인튜닝, 다음 교시)"
    else:
        label = "e5-small (학습 전)"
    labels[label] = row
    calls = row["expansion"]["llm_calls_per_query"] if row.get("expansion") else 0
    d = row["doc"]
    print(
        f"  {pad(label, 30)}{d['recall@1']:.3f}  {d['recall@5']:.3f}  {d['recall@10']:.3f}  {d['mrr@10']:.3f}   {calls}"
    )
print(
    f"  (질문 {table['n']}개라 한 질문이 R@5를 {1 / table['n']:.3f}씩 움직인다. 경향만 본다)"
)
r5 = {label: row["doc"]["recall@5"] for label, row in labels.items()}
base_r5, exp_r5 = r5.get("e5-small (학습 전)"), r5.get("e5-small + 쿼리 확장")
ft = next(((k, v) for k, v in r5.items() if "파인튜닝" in k), None)
if base_r5 is not None and exp_r5 is not None:
    print(
        f"  → 확장이 R@5를 {base_r5:.3f}에서 {exp_r5:.3f}로 바꿨다 (질문마다 LLM 1회)."
    )
if ft and exp_r5 is not None:
    print(
        f"  → 파인튜닝 모델은 LLM 호출 없이 {ft[1]:.3f}. 6교시에 test 전체로 다시 본다."
    )

base = json.loads((out_dir / "multilingual-e5-small.json").read_text(encoding="utf-8"))
expanded = json.loads(
    (out_dir / "multilingual-e5-small_expand.json").read_text(encoding="utf-8")
)
print("\n질문별 정답 순위 (원문 → 확장)")
docs = {d["id"]: d for d in load_corpus(CORPUS)}
better = worse = 0
for a, b in zip(base["per_question"], expanded["per_question"]):
    ra, rb = a["doc_rank"] or 999, b["doc_rank"] or 999
    mark = "↑" if rb < ra else ("↓" if rb > ra else "=")
    better += rb < ra
    worse += rb > ra
    print(
        f"  {mark} {rank_text(a['doc_rank']):>7} → {rank_text(b['doc_rank']):>7}  {a['query'][:44]}"
    )
print(
    f"좋아진 질문 {better}개 · 나빠진 질문 {worse}개 · 같은 질문 {len(questions) - better - worse}개"
)
hurt = next(
    (
        b
        for a, b in zip(base["per_question"], expanded["per_question"])
        if (b["doc_rank"] or 999) > (a["doc_rank"] or 999)
    ),
    None,
)
if hurt:
    top1 = docs.get(hurt["top10"][0], {}).get("title", "-") if hurt["top10"] else "-"
    print(
        f"\n나빠진 예: {hurt['query']}\n  검색어: {hurt['expanded_query']}\n  1위: {top1}"
    )
    print("  확장어에 든 법령 이름이 정답 법령과 다르면 검색이 그 법령 쪽으로 끌려간다")


# ============================================================
# 4. 정리: 학습 없는 선택지 표
# ============================================================
section("4. 학습 없는 선택지 (test 2,526개 기준)")
full = (
    ROOT
    / "experiments"
    / "results"
    / "lecture"
    / "05_evaluate"
    / "multilingual-e5-small_test.json"
)
if full.exists():
    r = json.loads(full.read_text(encoding="utf-8"))
    e5_row = f"{r['doc']['recall@5']:.3f}  {r['doc']['recall@10']:.3f}"
else:
    e5_row = "01_metrics.py를 먼저 실행"
print("  " + pad("선택지", 24) + pad("R@5    R@10", 26) + "질문당 LLM 호출")
print("  " + pad("e5-small", 24) + pad(e5_row, 26) + "0")
print(
    "  "
    + pad("e5-small + 쿼리 확장", 24)
    + pad(f"측정 전 (캐시 {len(cached)}/{len(test):,})", 26)
    + "1"
)
print(
    "\n확장 행은 무료 한도 때문에 비워 둔다. 질문 몇 개로 본 경향과 질문당 비용만 가지고 판단한다."
)
print(
    "다음 시간(6교시): 작은 모델을 한 번 학습해 이 비용을 '학습 1회'로 옮기면 얼마나 오르나"
)
print(f"\n결과: {rel(out_dir)}")
