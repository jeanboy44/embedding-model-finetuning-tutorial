"""
실습 1-1 (5교시): 코퍼스 — 법령 조문을 검색 문서로
==================================================

학습 목표:
- 받은 코퍼스(data/processed/law_docs.json)의 규모를 숫자로 본다: 법령 · 조문 · 검색 문서 · 테마
- 검색 문서 한 건이 어떤 필드로 되어 있는지 본다 (id, 제목, 본문, 법령, 테마, 상위 조 id, 항·호)
- 모델(e5)이 512토큰까지만 읽기 때문에 긴 조문을 항·호 단위로 나눴다는 것을 길이 분포로 확인한다

무엇을 하나 (기본 모드, 20초 안쪽):
    1. 규모: 법령 · 조문 · 검색 문서 수를 세고 테마별로 표를 찍는다 (ragkit.data.load_corpus)
    2. 검색 문서 한 건: 장표 예시 조문의 필드를 그대로 찍고, 임베딩하는 글의 모양을 보인다
    3. 항·호 분할: parent_id가 같은 문서를 세어 나뉜 조문 수 · 조각 수 · 예시 조문을 보인다
    4. 길이 분포: 글자 수와 e5 토큰 수의 분위수, 512토큰 넘는 문서 비율을 나누기 전후로 비교한다
    5. (--run) 전기 테마만 임시 폴더에 직접 만들어 받은 코퍼스와 id · 본문을 비교한다

2022: 법령 사이트에서 조문을 긁어 엑셀로 정리하고, 긴 조문은 사람이 잘랐다
2026: 법령이 Git(legalize-kr)에 있고, 스크립트 하나로 받아 나눈다

사전 준비:
    uv run python scripts/data_version.py pull v1      # data/processed/law_docs.json (강사 Drive)
    토큰 길이를 재려면 extra [torch] (uv sync --all-packages --all-extras). 없으면 글자 수만 본다

실행:
    uv run python lecture/05_data_eval/01_corpus.py           # 받은 코퍼스 확인 (몇 초)
    uv run python lecture/05_data_eval/01_corpus.py --run     # 전기(electric) 테마만 직접 받아 만들어 비교
                                                         # (임시 폴더, 인터넷 필요, 수 초)
"""

import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cyclopts
from cyclopts import Parameter
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

os.environ.setdefault(
    "TRANSFORMERS_VERBOSITY", "error"
)  # 512 넘는 문장 경고는 아래에서 직접 센다

from ragkit.config import get_settings
from ragkit.data import doc_text, load_corpus
from ragkit.embeddings import format_passages

console = Console(emoji=False)
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
    console.print()
    console.rule(f"[bold]{escape(title)}[/]", style="blue")


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


@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(
        False, description=f"{RUN_THEME} 테마 코퍼스를 임시 폴더에 직접 만들어 비교"
    )


def cli(*, options: Options | None = None) -> Options:
    """실습 1-1: 코퍼스"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)

if not CORPUS.exists():
    console.print(
        Panel(
            f"코퍼스가 없습니다: {escape(str(CORPUS))}\n"
            "받기: [dim]uv run python scripts/data_version.py pull v1[/]\n"
            "직접 만들기(전체, 수 분): [dim]uv run python scripts/prepare_law_data.py[/]",
            title="필요한 산출물이 없습니다",
            border_style="red",
        )
    )
    sys.exit(1)

docs = load_corpus(CORPUS)


# ============================================================
# 1. 규모
# ============================================================
# 코퍼스 전체를 돌며 법령(category) · 조문(parent_id) · 검색 문서(id) 수를 센다.
# 법령 하나는 법률 · 시행령 · 시행규칙 파일 여러 개로 되어 있어 파일 수도 같이 보인다.
# 출력에서 볼 것: 법령 54개 · 조문 11,723개 · 검색 문서 25,967개, 테마별 크기 차이.
section("1. 규모: 법령 · 조문 · 검색 문서")
laws = {d["category"] for d in docs}
# law_name(파일)마다 하나씩만 세어 법률 · 시행령 · 시행규칙 파일 수를 구한다
law_files = Counter(d["law_type"] for d in {d["law_name"]: d for d in docs}.values())
articles = {d["parent_id"] for d in docs}  # parent_id = 나누기 전 조문의 id
console.print(
    f"법령       [bold]{len(laws):>7,}[/]개  (법률·시행령·시행규칙 파일 {sum(law_files.values())}개: "
    + ", ".join(f"{escape(t)} {n}" for t, n in law_files.most_common())
    + ")",
    highlight=False,
)
console.print(f"조문       [bold]{len(articles):>7,}[/]개", highlight=False)
console.print(
    f"검색 문서  [bold]{len(docs):>7,}[/]개  ← 긴 조문을 항·호로 나눈 조각",
    highlight=False,
)

console.print("\n테마별")
by_theme: dict[str, list[dict]] = {}
for d in docs:
    by_theme.setdefault(d["theme"], []).append(d)
theme_table = Table(box=box.SIMPLE_HEAD, show_edge=False)
theme_table.add_column("테마")
theme_table.add_column("법령", justify="right")
theme_table.add_column("검색 문서", justify="right")
for theme, items in sorted(by_theme.items(), key=lambda x: -len(x[1])):  # 큰 테마부터
    n_laws = len({d["category"] for d in items})
    theme_table.add_row(escape(theme), f"{n_laws}개", f"{len(items):,}개")
console.print(theme_table)


# ============================================================
# 2. 검색 문서 한 건
# ============================================================
# 장표에 나온 최저임금법 제5조를 골라 검색 문서의 필드를 하나씩 찍는다.
# 모델이 읽는 글은 ragkit.data.doc_text(제목 + 줄바꿈 + 본문)이고, "passage: " 접두어는
# ragkit.embeddings.format_passages가 코드에서 붙인다 (데이터에는 없다).
section("2. 검색 문서 한 건 (장표의 예시)")
# 예시 조문이 없으면 첫 문서
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
    console.print(f"  {field:12s} {escape(repr(example[field]))}")
console.print(
    f"  {'text':12s} {example['text'][:160]}...", markup=False, highlight=False
)  # 본문은 앞 160자만
console.print(
    "\n임베딩하는 글: 제목 + 줄바꿈 + 본문 (ragkit.data.doc_text), 앞에 'passage: '를 붙인다"
)


# ============================================================
# 3. 항·호 분할
# ============================================================
# parent_id가 같은 검색 문서 수를 세어, 조문 하나가 몇 조각으로 나뉘었는지 본다.
# 출력에서 볼 것: 나뉜 조문 2,596개 → 조각 16,840개, 최대 32조각,
# 예시 전기사업법 제2조(정의)는 호를 묶어 5조각이고 조각마다 머리말을 되풀이한다.
section("3. 긴 조문은 항·호 단위로 나눈다")
pieces = Counter(d["parent_id"] for d in docs)  # 조문 id → 조각 수
split_articles = [pid for pid, n in pieces.items() if n > 1]  # 2조각 이상 = 나뉜 조문
console.print(
    f"나뉜 조문 [bold]{len(split_articles):,}[/]개 / {len(articles):,}개"
    f"  → 조각 [bold]{sum(pieces[p] for p in split_articles):,}[/]개",
    highlight=False,
)
console.print(f"조문 하나가 가장 많이 나뉜 경우: {max(pieces.values())}조각")
console.print(f"\n예: {escape(SPLIT_EXAMPLE)}")
piece_table = Table(box=box.SIMPLE_HEAD, show_edge=False)
piece_table.add_column("조각 id")
piece_table.add_column("글자 수", justify="right")
for d in [d for d in docs if d["parent_id"] == SPLIT_EXAMPLE][:6]:  # 앞 6조각까지만
    piece_table.add_row(escape(d["id"]), f"{len(doc_text(d)):,}자")
console.print(piece_table)
console.print(
    '  조각마다 같은 조의 머리말("이 법에서 사용하는 용어의 뜻은...")을 앞에 붙여 문맥을 남긴다'
)


# ============================================================
# 4. 길이 분포
# ============================================================
# 검색 문서의 글자 수와 e5 토크나이저로 잰 토큰 수("passage: " 포함)의 분위수를 찍는다.
# 토큰 수 재기는 약 10초. 나누기 전 조 단위 코퍼스(law_docs_article_level.json)가 있으면
# 512토큰 넘는 비율을 전후로 비교한다. 출력에서 볼 것: 18.6% → 1.0%, 가장 긴 문서.
section(f"4. 길이 분포: 모델은 {MAX_TOKENS}토큰까지만 읽는다")
chars = [len(doc_text(d)) for d in docs]
console.print(f"글자 수   {percentiles(chars)}")
start = time.perf_counter()
tokens = token_lengths(docs)
if tokens is None:
    console.print(
        "[yellow]토큰 수는 extra \\[torch]가 있어야 잰다[/] (uv sync --all-packages --all-extras)"
    )
else:
    over = sum(t > MAX_TOKENS for t in tokens)  # 모델이 뒷부분을 못 읽는 문서 수
    console.print(
        f"토큰 수   {percentiles(tokens)}  [dim]({time.perf_counter() - start:.1f}초)[/]"
    )
    console.print(
        f"  {MAX_TOKENS}토큰 넘는 검색 문서 [bold]{over:,}[/]개 ({over / len(tokens):.1%})",
        highlight=False,
    )
    longest = max(range(len(docs)), key=lambda i: tokens[i])
    console.print(
        f"  가장 긴 문서: {escape(docs[longest]['title'])} ({tokens[longest]:,}토큰)"
        " — 표처럼 항·호로 더 나눌 수 없는 문서가 남는다",
        highlight=False,
    )
    if ARTICLE_LEVEL.exists():  # 나누기 전 코퍼스가 있을 때만 전후 비교
        before = load_corpus(ARTICLE_LEVEL)
        before_tokens = token_lengths(before) or []
        before_over = sum(t > MAX_TOKENS for t in before_tokens)
        console.print(
            f"  나누기 전(조 단위 {len(before):,}개)에는 [bold]{before_over:,}[/]개"
            f" ({before_over / max(len(before), 1):.1%})가 넘었다 → 뒷부분은 모델이 못 읽는다",
            highlight=False,
        )


# ============================================================
# 5. (--run) 직접 만들어 보기
# ============================================================
# scripts/prepare_law_data.py를 하위 프로세스로 불러 전기 테마만 임시 폴더에 만든다
# (legalize-kr을 받으므로 인터넷 필요, 수 초). 받은 코퍼스의 electric 문서와
# id 집합 · 본문이 같은지 비교한다. 다르면 그 사이 법령이 개정된 것이다.
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
        console.print(
            "[dim]$ uv run python scripts/prepare_law_data.py --themes "
            f"{escape(RUN_THEME)} --repo-dir <임시 폴더> --output <임시 폴더>/law_docs.json[/]"
        )
        start = time.perf_counter()
        sys.stdout.flush()  # 아래 명령의 출력과 순서가 섞이지 않게
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0 or not out.exists():
            console.print(
                "\n[red]만들지 못했습니다.[/] 인터넷(github.com) 연결을 확인하세요. 받은 코퍼스는 그대로다."
            )
        else:
            mine = load_corpus(out)
            given = [d for d in docs if d["theme"] == RUN_THEME]
            # 문서 구성(id 집합)이 같은가
            same_ids = {d["id"] for d in mine} == {d["id"] for d in given}
            given_by_id = {d["id"]: d for d in given}
            same_text = sum(
                given_by_id.get(d["id"], {}).get("text") == d["text"] for d in mine
            )
            console.print(
                f"\n[bold]{time.perf_counter() - start:.1f}초[/]. 직접 만든 문서 [bold]{len(mine):,}[/]개"
                f" vs 받은 코퍼스의 {escape(RUN_THEME)} {len(given):,}개",
                highlight=False,
            )
            console.print(
                f"  id 집합 같음: {'예' if same_ids else '아니오'},"
                f" 본문까지 같은 문서 {same_text:,}개"
            )
            console.print(
                "  다르면 그 사이 법령이 개정된 것이다 (legalize-kr은 최신 법령을 받는다."
                " 시점을 고정하려면 --as-of YYYY-MM-DD)"
            )
    console.print("[dim]임시 폴더는 지웠다. data/processed/law_docs.json은 바뀌지 않았다.[/]")

console.print("""
정리:
- 검색 단위는 법령이 아니라 조문, 긴 조문은 다시 항·호로 나눈 조각이다 (숫자는 위 출력)
- 긴 조문을 나누지 않으면 512토큰 뒤의 내용은 임베딩에 들어가지 않는다
- 다음: [dim]uv run python lecture/05_data_eval/02_split.py[/] (조 단위 분할)
""")
