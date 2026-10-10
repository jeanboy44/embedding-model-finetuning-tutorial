"""
실습 1-1 (4교시): 코퍼스 — 법령 조문을 검색 문서로
==================================================

학습 목표:
- 받은 코퍼스(data/processed/law_docs.json)의 규모를 숫자로 본다: 법령 · 조문 · 검색 문서 · 테마
- 검색 문서 한 건이 어떤 필드로 되어 있는지 본다 (id, 제목, 본문, 법령, 테마, 상위 조 id, 항·호)
- 모델(e5)이 512토큰까지만 읽기 때문에 긴 조문을 항·호 단위로 나눴다는 것을 길이 분포로 확인한다

2022: 법령 사이트에서 조문을 긁어 엑셀로 정리하고, 긴 조문은 사람이 잘랐다
2026: 법령이 Git(legalize-kr)에 있고, 스크립트 하나로 받아 나눈다

사전 준비:
    uv run python scripts/data_version.py pull v1      # data/processed/law_docs.json (강사 Drive)
    토큰 길이를 재려면 extra [torch] (uv sync --all-packages --all-extras). 없으면 글자 수만 본다

실행:
    uv run python lecture/04_data/01_corpus.py           # 받은 코퍼스 확인 (몇 초)
    uv run python lecture/04_data/01_corpus.py --run     # 전기(electric) 테마만 직접 받아 만들어 비교
                                                         # (임시 폴더, 인터넷 필요, 수 초)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault(
    "TRANSFORMERS_VERBOSITY", "error"
)  # 512 넘는 문장 경고는 아래에서 직접 센다

from ragkit.config import get_settings
from ragkit.data import doc_text, load_corpus
from ragkit.embeddings import format_passages

ROOT = Path(__file__).resolve().parents[2]
settings = get_settings()
CORPUS = settings.data_dir / "processed" / "law_docs.json"
ARTICLE_LEVEL = (
    settings.data_dir / "processed" / "law_docs_article_level.json"
)  # 나누기 전
MAX_TOKENS = 512  # e5-small이 읽는 최대 토큰 수
RUN_THEME = "electric"  # --run에서 직접 만들 테마 (가장 작다)
EXAMPLE_ID = "최저임금법_법률_제5조"  # 장표의 예시 (수습 근로자 최저임금 감액)
SPLIT_EXAMPLE = "전기사업법_법률_제2조"  # 정의 조문: 호가 많아 여러 조각으로 나뉜다


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def percentiles(values: list[int]) -> str:
    ordered = sorted(values)
    pick = {
        p: ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))]
        for p in (50, 90, 99)
    }
    return f"중앙값 {pick[50]:,} · 90% {pick[90]:,} · 99% {pick[99]:,} · 최대 {ordered[-1]:,}"


def token_lengths(docs: list[dict]) -> list[int] | None:
    """e5 토크나이저로 잰 검색 문서 길이 ("passage: " 포함). torch extra가 없으면 None."""
    try:
        from ragkit.models import load_tokenizer

        tokenizer = load_tokenizer(settings.embedding_model_name)
    except ImportError:
        return None
    texts = format_passages([doc_text(d) for d in docs])
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]]


parser = argparse.ArgumentParser(description="실습 1-1: 코퍼스")
parser.add_argument(
    "--run",
    action="store_true",
    help=f"{RUN_THEME} 테마 코퍼스를 임시 폴더에 직접 만들어 비교",
)
args = parser.parse_args()

if not CORPUS.exists():
    print(f"코퍼스가 없습니다: {CORPUS}")
    print("받기: uv run python scripts/data_version.py pull v1")
    print("직접 만들기(전체, 수 분): uv run python scripts/prepare_law_data.py")
    sys.exit(1)

docs = load_corpus(CORPUS)


# ============================================================
# 1. 규모
# ============================================================
section("1. 규모: 법령 · 조문 · 검색 문서")
laws = {d["category"] for d in docs}
law_files = Counter(d["law_type"] for d in {d["law_name"]: d for d in docs}.values())
articles = {d["parent_id"] for d in docs}
print(
    f"법령       {len(laws):>7,}개  (법률·시행령·시행규칙 파일 {sum(law_files.values())}개: "
    + ", ".join(f"{t} {n}" for t, n in law_files.most_common())
    + ")"
)
print(f"조문       {len(articles):>7,}개")
print(f"검색 문서  {len(docs):>7,}개  ← 긴 조문을 항·호로 나눈 조각")

print("\n테마별")
by_theme: dict[str, list[dict]] = {}
for d in docs:
    by_theme.setdefault(d["theme"], []).append(d)
for theme, items in sorted(by_theme.items(), key=lambda x: -len(x[1])):
    n_laws = len({d["category"] for d in items})
    print(f"  {theme:10s} 법령 {n_laws:3d}개  검색 문서 {len(items):6,}개")


# ============================================================
# 2. 검색 문서 한 건
# ============================================================
section("2. 검색 문서 한 건 (장표의 예시)")
example = next((d for d in docs if d["parent_id"] == EXAMPLE_ID), docs[0])
for field in (
    "id",
    "title",
    "category",
    "theme",
    "law_type",
    "chapter",
    "parent_id",
    "paragraph",
):
    print(f"  {field:12s} {example[field]!r}")
print(f"  {'text':12s} {example['text'][:160]}...")
print(
    "\n임베딩하는 글: 제목 + 줄바꿈 + 본문 (ragkit.data.doc_text), 앞에 'passage: '를 붙인다"
)


# ============================================================
# 3. 항·호 분할
# ============================================================
section("3. 긴 조문은 항·호 단위로 나눈다")
pieces = Counter(d["parent_id"] for d in docs)
split_articles = [pid for pid, n in pieces.items() if n > 1]
print(
    f"나뉜 조문 {len(split_articles):,}개 / {len(articles):,}개"
    f"  → 조각 {sum(pieces[p] for p in split_articles):,}개"
)
print(f"조문 하나가 가장 많이 나뉜 경우: {max(pieces.values())}조각")
print(f"\n예: {SPLIT_EXAMPLE}")
for d in [d for d in docs if d["parent_id"] == SPLIT_EXAMPLE][:6]:
    print(f"  {d['id']:40s} {len(doc_text(d)):5,}자")
print(
    '  조각마다 같은 조의 머리말("이 법에서 사용하는 용어의 뜻은...")을 앞에 붙여 문맥을 남긴다'
)


# ============================================================
# 4. 길이 분포
# ============================================================
section(f"4. 길이 분포: 모델은 {MAX_TOKENS}토큰까지만 읽는다")
chars = [len(doc_text(d)) for d in docs]
print(f"글자 수   {percentiles(chars)}")
start = time.perf_counter()
tokens = token_lengths(docs)
if tokens is None:
    print("토큰 수는 extra [torch]가 있어야 잰다 (uv sync --all-packages --all-extras)")
else:
    over = sum(t > MAX_TOKENS for t in tokens)
    print(f"토큰 수   {percentiles(tokens)}  ({time.perf_counter() - start:.1f}초)")
    print(f"  {MAX_TOKENS}토큰 넘는 검색 문서 {over:,}개 ({over / len(tokens):.1%})")
    longest = max(range(len(docs)), key=lambda i: tokens[i])
    print(
        f"  가장 긴 문서: {docs[longest]['title']} ({tokens[longest]:,}토큰)"
        " — 표처럼 항·호로 더 나눌 수 없는 문서가 남는다"
    )
    if ARTICLE_LEVEL.exists():
        before = load_corpus(ARTICLE_LEVEL)
        before_tokens = token_lengths(before) or []
        before_over = sum(t > MAX_TOKENS for t in before_tokens)
        print(
            f"  나누기 전(조 단위 {len(before):,}개)에는 {before_over:,}개"
            f" ({before_over / max(len(before), 1):.1%})가 넘었다 → 뒷부분은 모델이 못 읽는다"
        )


# ============================================================
# 5. (--run) 직접 만들어 보기
# ============================================================
if args.run:
    section(f"5. --run: {RUN_THEME} 테마 코퍼스를 직접 만든다")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "law_docs.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "prepare_law_data.py"),
            "--themes",
            RUN_THEME,
            "--repo-dir",
            str(Path(tmp) / "legalize-kr"),
            "--output",
            str(out),
        ]
        print(
            "$ uv run python scripts/prepare_law_data.py --themes",
            RUN_THEME,
            "--repo-dir <임시 폴더> --output <임시 폴더>/law_docs.json",
        )
        start = time.perf_counter()
        sys.stdout.flush()  # 아래 명령의 출력과 순서가 섞이지 않게
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0 or not out.exists():
            print(
                "\n만들지 못했습니다. 인터넷(github.com) 연결을 확인하세요. 받은 코퍼스는 그대로다."
            )
        else:
            mine = load_corpus(out)
            given = [d for d in docs if d["theme"] == RUN_THEME]
            same_ids = {d["id"] for d in mine} == {d["id"] for d in given}
            given_by_id = {d["id"]: d for d in given}
            same_text = sum(
                given_by_id.get(d["id"], {}).get("text") == d["text"] for d in mine
            )
            print(
                f"\n{time.perf_counter() - start:.1f}초. 직접 만든 문서 {len(mine):,}개"
                f" vs 받은 코퍼스의 {RUN_THEME} {len(given):,}개"
            )
            print(
                f"  id 집합 같음: {'예' if same_ids else '아니오'},"
                f" 본문까지 같은 문서 {same_text:,}개"
            )
            print(
                "  다르면 그 사이 법령이 개정된 것이다 (legalize-kr은 최신 법령을 받는다."
                " 시점을 고정하려면 --as-of YYYY-MM-DD)"
            )
    print("임시 폴더는 지웠다. data/processed/law_docs.json은 바뀌지 않았다.")

print("""
정리:
- 검색 단위는 법령이 아니라 조문, 긴 조문은 다시 항·호로 나눈 조각이다 (숫자는 위 출력)
- 긴 조문을 나누지 않으면 512토큰 뒤의 내용은 임베딩에 들어가지 않는다
- 다음: uv run python lecture/04_data/02_questions.py (이 문서에 대한 질문 데이터)
""")
