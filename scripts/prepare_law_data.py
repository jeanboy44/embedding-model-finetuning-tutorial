"""legalize-kr 저장소에서 법령을 내려받아 조문 단위 코퍼스(JSON)로 만든다.

legalize-kr(https://github.com/legalize-kr/legalize-kr)은 대한민국 법령을 Markdown으로,
개정 이력을 실제 공포일자를 가진 Git 커밋으로 관리한다. 필요한 법령 폴더만 sparse checkout으로 받는다.

사용법:
    uv run python scripts/prepare_law_data.py                       # 전체 테마, 최신 법령
    uv run python scripts/prepare_law_data.py --themes youth        # 사회 첫걸음 테마만
    uv run python scripts/prepare_law_data.py --as-of 2024-01-01    # 특정 날짜 시점의 법령

출력 (기본값):
    data/raw/legalize-kr/            Git 저장소 (sparse checkout)
    data/processed/law_docs.json     조문 목록 (sample_docs.json과 같은 id/title/text/category + 메타데이터)
"""

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from cyclopts import run

from ragkit.config import get_settings

REPO_URL = "https://github.com/legalize-kr/legalize-kr.git"

# 테마별 법령 폴더 이름 (kr/<폴더>/). 폴더 안의 법률·시행령·시행규칙을 모두 사용한다.
THEMES: dict[str, list[str]] = {
    "electric": [
        "전기사업법",
        "전기안전관리법",
        "전기공사업법",
        "전기용품및생활용품안전관리법",
        "전기산업발전기본법",
        "농어촌전기공급사업촉진법",
        "전기공사공제조합법",
    ],
    "youth": [
        "근로기준법",
        "최저임금법",
        "기간제및단시간근로자보호등에관한법률",
        "근로자퇴직급여보장법",
        "남녀고용평등과일ㆍ가정양립지원에관한법률",
        "채용절차의공정화에관한법률",
        "청년고용촉진특별법",
        "청년기본법",
        "고용보험법",
        "취업후학자금상환특별법",
        "한국장학재단설립등에관한법률",
        "주택임대차보호법",
    ],
    "traffic": [
        "도로교통법",
        "교통사고처리특례법",
        "자동차관리법",
        "자동차손해배상보장법",
        "여객자동차운수사업법",
        "화물자동차운수사업법",
        "도로법",
        "교통안전법",
        "주차장법",
        "자동차등록령",
    ],
    "tax": [
        "소득세법",
        "부가가치세법",
        "조세특례제한법",
        "국세기본법",
        "상속세및증여세법",
        "종합부동산세법",
        "지방세법",
    ],
    "finance": [
        "자본시장과금융투자업에관한법률",
        "은행법",
        "전자금융거래법",
        "가상자산이용자보호등에관한법률",
        "금융소비자보호에관한법률",
        "예금자보호법",
        "여신전문금융업법",
        "대부업등의등록및금융이용자보호에관한법률",
        "신용정보의이용및보호에관한법률",
        "보험업법",
    ],
    "consumer": [
        "전자상거래등에서의소비자보호에관한법률",
        "할부거래에관한법률",
        "약관의규제에관한법률",
        "표시ㆍ광고의공정화에관한법률",
        "소비자기본법",
        "독점규제및공정거래에관한법률",
        "방문판매등에관한법률",
        "상가건물임대차보호법",
    ],
}

ARTICLE_RE = re.compile(r"^##### (제\d+조(?:의\d+)?)(?:\s*\((.+)\))?\s*$")
CHAPTER_RE = re.compile(r"^## (제\d+장(?:의\d+)?.*?)(?:\s*<.*>)?\s*$")
# 본문이 "삭제 <2008.3.21>"뿐인 조문, 법령 자체를 폐지하는 조문은 코퍼스에서 제외한다.
EMPTY_BODY_RE = re.compile(r"^(삭제\s*<[^>]*>|.*이를 폐지한다\.?)$")
# 검색에 잡음인 개정 이력(<개정 2018.3.20>, <신설 …>, <2020.3.31>)과 이미지 태그를 본문에서 지운다.
NOISE_RE = re.compile(r"\s*<(?:개정|신설|\d{4}\.)[^>]*>|<img[^>]*>|</img>")
# 삭제된 항·호·목 줄(개정 태그를 지운 뒤 "② 삭제", "3. 삭제", "가. 삭제"만 남은 줄)은 본문에서 지운다.
DELETED_LINE_RE = re.compile(r"^(?:[①-⑳㉑-㉟]|\d+(?:의\d+)?\.|[가-하]\.)\s*삭제\s*$")
# 제16항부터는 원문자 대신 `**<16>** <16> 본문`처럼 쓴다. 표기를 `<16> 본문` 하나로 줄인다.
ANGLE_PARAGRAPH_RE = re.compile(r"^\*\*<(\d+)>\*\*\s*(?:<\1>\s*)?", re.M)
# 항 번호(①~⑳, ㉑~㉟, <16> 등)와 호 번호(1. 2의3.)는 줄 맨 앞에 온다.
PARAGRAPH_RE = re.compile(r"^(?:[①-⑳㉑-㉟]|<\d+>)", re.M)
ITEM_RE = re.compile(r"^(\d+(?:의\d+)?)\.\s", re.M)
# 조문 하나(제목 + 본문)의 최대 글자 수. 넘으면 항 단위로, 항도 넘으면 호 묶음으로 나눈다.
# 임베딩 모델(multilingual-e5, 최대 512토큰)에서 잘리지 않게 잡은 값이다. 한국어 법령은 약 1.8자/토큰이다.
MAX_CHARS = 800


def _paragraph_no(mark: str) -> int:
    """항 번호 기호(①, ⑪, ㉑, <16>)를 숫자로 바꾼다."""
    if mark.startswith("<"):
        return int(mark[1:-1])
    code = ord(mark)
    return code - 0x2460 + 1 if code <= 0x2473 else code - 0x3251 + 21


def _split_items(text: str, heading_chars: int, max_chars: int) -> list[tuple[str, str]]:
    """호 목록을 앞에서부터 한도 안으로 묶는다. 머리 문장(첫 호 앞)은 묶음마다 반복한다."""
    matches = list(ITEM_RE.finditer(text))
    if len(matches) < 2:
        return [("", text)]
    lead = text[: matches[0].start()].strip()
    ends = [m.start() for m in matches[1:]] + [len(text)]
    items = [(m.group(1), text[m.start() : end].strip()) for m, end in zip(matches, ends)]

    groups: list[list[tuple[str, str]]] = [[]]
    size = heading_chars + len(lead)
    for item in items:
        length = len(item[1]) + 1
        if groups[-1] and size + length > max_chars:
            groups.append([])
            size = heading_chars + len(lead)
        groups[-1].append(item)
        size += length

    chunks = []
    for group in groups:
        first, last = group[0][0], group[-1][0]
        label = f"제{first}호" if first == last else f"제{first}~{last}호"
        lines = [lead] if lead else []
        chunks.append((label, "\n".join(lines + [item for _, item in group])))
    return chunks


def split_article(
    text: str, heading_chars: int, max_chars: int | None = None
) -> list[tuple[str, str]]:
    """긴 조문을 항 단위로, 항이 여전히 길면 호 묶음으로 나눈다.

    Args:
        text: 조문 본문.
        heading_chars: 조각마다 붙는 제목의 글자 수.
        max_chars: 제목 + 본문의 최대 글자 수. 기본값은 MAX_CHARS.

    Returns:
        [(레이블, 본문), ...]. 나누지 않으면 레이블은 빈 문자열이다.
        레이블 예: "제3항", "제1~12호", "제1항 제1~12호".
    """
    max_chars = max_chars or MAX_CHARS
    if heading_chars + len(text) <= max_chars:
        return [("", text)]

    matches = list(PARAGRAPH_RE.finditer(text))
    starts, marks = [m.start() for m in matches], [m.group() for m in matches]
    if len(starts) >= 2:
        # 첫 항 앞에 글이 있으면(드묾) 첫 항에 붙인다.
        begins, ends = [0, *starts[1:]], [*starts[1:], len(text)]
        paragraphs = [
            (f"제{_paragraph_no(mark)}항", text[begin:end].strip())
            for mark, begin, end in zip(marks, begins, ends)
        ]
    else:
        paragraphs = [("", text)]

    chunks: list[tuple[str, str]] = []
    for label, body in paragraphs:
        if heading_chars + len(label) + len(body) <= max_chars:
            chunks.append((label, body))
            continue
        for item_label, item_body in _split_items(body, heading_chars + len(label), max_chars):
            chunks.append((f"{label} {item_label}".strip(), item_body))
    return chunks


def clean_text(text: str) -> str:
    """Markdown 강조·이스케이프, 개정 이력 태그, 이미지 태그, 삭제된 항·호·목 줄을 지운다."""
    text = ANGLE_PARAGRAPH_RE.sub(r"<\1> ", text)
    text = NOISE_RE.sub("", text.replace("**", "").replace("\\.", "."))
    lines = [line for line in text.splitlines() if not DELETED_LINE_RE.match(line.strip())]
    return "\n".join(lines).strip()


@dataclass
class LawArticle:
    """조문 하나(긴 조문은 그 조각 하나). id/title/text/category는 data/sample_docs.json과 같은 형식이다.

    긴 조문을 나눈 조각은 id와 title 끝에 paragraph 레이블(예: "제3항", "제1~12호")이 붙고,
    parent_id가 원래 조문 id를 가리킨다. 나누지 않은 조문은 paragraph가 빈 문자열이고 parent_id == id다.
    """

    id: str
    title: str
    text: str
    category: str
    theme: str
    law_name: str
    law_type: str
    article_no: str
    article_title: str
    chapter: str
    promulgation_date: str
    effective_date: str
    source_url: str
    parent_id: str
    paragraph: str


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=off", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sync_repo(repo_dir: Path, law_dirs: list[str], *, as_of: str | None = None) -> str:
    """legalize-kr 저장소를 받아 지정한 법령 폴더만 checkout한다.

    Args:
        repo_dir: 저장소를 둘 로컬 경로.
        law_dirs: checkout할 kr/ 아래 법령 폴더 이름 목록.
        as_of: "YYYY-MM-DD". 주어지면 그 날짜 이전 마지막 커밋 시점으로 checkout한다.
            None이면 최신 main.

    Returns:
        checkout한 커밋 해시.
    """
    if not (repo_dir / ".git").exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        # 최신본만 필요하면 커밋 1개만, 과거 시점이 필요하면 전체 이력(트리·파일 내용은 필요할 때만)을 받는다.
        history = ["--filter=tree:0"] if as_of else ["--depth=1", "--filter=blob:none"]
        print(f"클론 중: {REPO_URL} → {repo_dir}")
        _git("clone", *history, "--no-checkout", "--sparse", REPO_URL, str(repo_dir))
    elif as_of and _git("rev-parse", "--is-shallow-repository", cwd=repo_dir) == "true":
        print("과거 시점을 찾기 위해 전체 커밋 이력을 받는 중...")
        _git("fetch", "--unshallow", "--filter=tree:0", "origin", "main", cwd=repo_dir)
    else:
        _git(
            "fetch",
            "--depth=1" if not as_of else "--filter=tree:0",
            "origin",
            "main",
            cwd=repo_dir,
        )

    _git(
        "sparse-checkout",
        "set",
        "--no-cone",
        *[f"/kr/{d}/" for d in law_dirs],
        cwd=repo_dir,
    )

    if as_of:
        commit = _git(
            "rev-list", "-1", f"--before={as_of} 23:59:59", "origin/main", cwd=repo_dir
        )
        if not commit:
            raise ValueError(f"{as_of} 이전 커밋이 없습니다.")
    else:
        commit = _git("rev-parse", "origin/main", cwd=repo_dir)
    _git("checkout", "--quiet", "--detach", commit, cwd=repo_dir)
    return commit


def parse_law_file(path: Path) -> tuple[dict, list[tuple[str, str, str, str]]]:
    """법령 Markdown 파일을 frontmatter와 조문 목록으로 나눈다.

    부칙(## 부칙 이후)과 삭제·폐지된 조문은 제외한다.

    Returns:
        (frontmatter 딕셔너리, [(조문번호, 조문제목, 장 제목, 본문), ...])
    """
    content = path.read_text(encoding="utf-8")
    _, front, body = content.split("---\n", 2)
    meta = yaml.safe_load(front)

    articles: list[tuple[str, str, str, str]] = []
    chapter = ""
    current: tuple[str, str, str] | None = None
    lines: list[str] = []

    def flush() -> None:
        if current is None:
            return
        raw = "\n".join(line.strip() for line in lines if line.strip())
        if raw and not EMPTY_BODY_RE.match(raw) and (text := clean_text(raw)):
            articles.append((*current, text))

    for line in body.splitlines():
        if line.startswith("## 부칙"):
            break
        if chapter_match := CHAPTER_RE.match(line):
            chapter = chapter_match.group(1).strip()
        elif article_match := ARTICLE_RE.match(line):
            flush()
            current = (
                article_match.group(1),
                (article_match.group(2) or "").strip(),
                chapter,
            )
            lines = []
        elif current is not None and not line.startswith("#"):
            lines.append(line)
    flush()
    return meta, articles


def latest_law_files(law_dir: Path) -> list[Path]:
    """폴더 안에서 법령 제목별로 공포일자가 가장 최근인 파일만 고른다.

    같은 이름의 옛 법령이 `법률.md`, 현행 법령이 `법률(법률).md`로 함께 있는 경우가 있다
    (예: 근로기준법 — `법률.md`는 1997년 폐지 법률).
    """
    latest: dict[str, tuple[str, Path]] = {}
    for path in sorted(law_dir.glob("*.md")):
        meta, _ = parse_law_file(path)
        title, date = meta["제목"], str(meta["공포일자"])
        if title not in latest or date > latest[title][0]:
            latest[title] = (date, path)
    return [path for _, path in sorted(latest.values(), key=lambda item: item[1].name)]


def build_corpus(repo_dir: Path, themes: dict[str, list[str]]) -> list[LawArticle]:
    """checkout된 저장소에서 테마별 법령을 읽어 조문 목록을 만든다."""
    corpus: list[LawArticle] = []
    for theme, law_dirs in themes.items():
        for law_dir_name in law_dirs:
            law_dir = repo_dir / "kr" / law_dir_name
            if not law_dir.exists():
                print(
                    f"경고: {law_dir} 가 없습니다 (이 시점에 없던 법령일 수 있음). 건너뜁니다."
                )
                continue
            for path in latest_law_files(law_dir):
                meta, articles = parse_law_file(path)
                law_type = path.stem.split("(")[0]
                for article_no, article_title, chapter, text in articles:
                    heading = (
                        f"{article_no} ({article_title})"
                        if article_title
                        else article_no
                    )
                    title = f"{meta['제목']} {heading}"
                    parent_id = f"{law_dir_name}_{law_type}_{article_no}"
                    for paragraph, chunk in split_article(text, len(title)):
                        corpus.append(
                            LawArticle(
                                id=f"{parent_id}_{paragraph.replace(' ', '_')}"
                                if paragraph
                                else parent_id,
                                title=f"{title} {paragraph}" if paragraph else title,
                                text=chunk,
                                category=law_dir_name,
                                theme=theme,
                                law_name=meta["제목"],
                                law_type=law_type,
                                article_no=article_no,
                                article_title=article_title,
                                chapter=chapter,
                                promulgation_date=str(meta["공포일자"]),
                                effective_date=str(meta["시행일자"]),
                                source_url=meta["출처"],
                                parent_id=parent_id,
                                paragraph=paragraph,
                            )
                        )
    # 검색 인덱스(예: SQLite UNIQUE id)는 id가 고유해야 한다. 조각 나누기 규칙이 어긋나면 여기서 잡는다.
    counts: dict[str, int] = {}
    for article in corpus:
        counts[article.id] = counts.get(article.id, 0) + 1
    if duplicates := sorted(i for i, n in counts.items() if n > 1):
        raise ValueError(f"중복 id {len(duplicates)}개: {duplicates[:5]}")
    return corpus


def main(
    themes: list[str] | None = None,
    as_of: str | None = None,
    repo_dir: Path | None = None,
    output: Path | None = None,
) -> None:
    """legalize-kr 법령을 내려받아 조문 단위 코퍼스를 만든다.

    Args:
        themes: 사용할 테마 (electric, youth, traffic, tax, finance, consumer). 기본값은 전체.
        as_of: "YYYY-MM-DD". 해당 날짜 시점의 법령을 사용한다. 기본값은 최신.
        repo_dir: 저장소 경로. 기본값은 data/raw/legalize-kr.
        output: 출력 JSON 경로. 기본값은 data/processed/law_docs.json.
    """
    settings = get_settings()
    selected = {name: THEMES[name] for name in (themes or list(THEMES))}
    repo_dir = repo_dir or settings.data_dir / "raw" / "legalize-kr"
    output = output or settings.data_dir / "processed" / "law_docs.json"

    law_dirs = [d for dirs in selected.values() for d in dirs]
    commit = sync_repo(repo_dir, law_dirs, as_of=as_of)
    commit_date = _git("log", "-1", "--format=%cs", commit, cwd=repo_dir)
    print(f"checkout: {commit[:10]} ({commit_date})")

    corpus = build_corpus(repo_dir, selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(
            [asdict(article) for article in corpus], f, ensure_ascii=False, indent=2
        )

    for theme in selected:
        count = sum(article.theme == theme for article in corpus)
        print(f"  {theme}: 문서 {count}개")
    articles = len({article.parent_id for article in corpus})
    print(f"완료: 조문 {articles}개 → 문서 {len(corpus)}개 (긴 조문은 항·호 단위로 나눔) → {output}")


if __name__ == "__main__":
    run(main)
