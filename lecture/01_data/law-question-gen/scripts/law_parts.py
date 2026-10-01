"""법령 코퍼스를 법령·파트 단위로 보여 준다 (질문 생성 스킬용).

사용법:
    uv run python .claude/skills/law-question-gen/scripts/law_parts.py list
    uv run python .claude/skills/law-question-gen/scripts/law_parts.py list --theme youth
    uv run python .claude/skills/law-question-gen/scripts/law_parts.py show 최저임금법 --part 1

파트는 법령 안의 조문을 순서대로 약 PART_CHARS 글자씩 묶은 것이다. 같은 입력이면 항상 같은 파트로 나뉜다.
"""

import json
from pathlib import Path

from cyclopts import App

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CORPUS = PROJECT_ROOT / "data" / "processed" / "law_docs.json"
QUESTIONS_DIR = PROJECT_ROOT / "data" / "questions"
PART_CHARS = 40_000

# 검색 질문을 만들 가치가 낮은 조문 제목. 질문을 만들지 않거나 아주 적게 만든다.
LOW_VALUE_TITLES = {
    "규제의 재검토",
    "고유식별정보의 처리",
    "민감정보 및 고유식별정보의 처리",
    "권한의 위임",
    "권한의 위임ㆍ위탁",
    "업무의 위탁",
    "벌칙 적용에서 공무원 의제",
    "준용",
    "준용규정",
    "다른 법률과의 관계",
    "시행일",
}

app = App(help="법령 코퍼스를 법령·파트 단위로 보여 준다.")


def load_corpus(corpus: Path = CORPUS) -> list[dict]:
    with corpus.open(encoding="utf-8") as f:
        return json.load(f)


def split_parts(articles: list[dict], part_chars: int = PART_CHARS) -> list[list[dict]]:
    """조문을 순서대로 약 part_chars 글자씩 묶는다. 조문 하나가 둘로 쪼개지지는 않는다."""
    parts: list[list[dict]] = [[]]
    size = 0
    for article in articles:
        length = len(article["title"]) + len(article["text"])
        if parts[-1] and size + length > part_chars:
            parts.append([])
            size = 0
        parts[-1].append(article)
        size += length
    return parts


def law_parts(law: str, corpus: Path = CORPUS) -> list[list[dict]]:
    articles = [a for a in load_corpus(corpus) if a["category"] == law]
    if not articles:
        raise SystemExit(
            f"코퍼스에 '{law}' 법령이 없습니다. `list`로 이름을 확인하세요."
        )
    return split_parts(articles)


def output_path(law: str, part: int) -> Path:
    return QUESTIONS_DIR / f"{law}__p{part:02d}.jsonl"


@app.command(name="list")
def list_laws(theme: str | None = None, corpus: Path = CORPUS) -> None:
    """법령별 문서 수(긴 조문은 항·호 조각), 파트 수, 질문 생성 진행 상황을 보여 준다.

    Args:
        theme: 이 테마의 법령만 보여 준다 (electric, youth, traffic, tax, finance, consumer).
        corpus: 코퍼스 JSON 경로.
    """
    by_law: dict[str, list[dict]] = {}
    for article in load_corpus(corpus):
        if theme is None or article["theme"] == theme:
            by_law.setdefault(article["category"], []).append(article)

    total_parts = done_parts = 0
    print(f"{'theme':9s} {'문서':>5s} {'파트':>4s} {'완료':>4s}  법령")
    for law, articles in by_law.items():
        parts = split_parts(articles)
        done = sum(output_path(law, i).exists() for i in range(1, len(parts) + 1))
        total_parts += len(parts)
        done_parts += done
        print(
            f"{articles[0]['theme']:9s} {len(articles):5d} {len(parts):4d} {done:4d}  {law}"
        )
    print(f"합계: 법령 {len(by_law)}개, 파트 {done_parts}/{total_parts} 완료")


@app.command
def show(law: str, part: int = 1, corpus: Path = CORPUS) -> None:
    """한 법령의 한 파트에 들어 있는 조문을 모두 출력한다.

    Args:
        law: 법령 이름 (코퍼스의 category, 예: 최저임금법).
        part: 1부터 시작하는 파트 번호.
        corpus: 코퍼스 JSON 경로.
    """
    parts = law_parts(law, corpus)
    if not 1 <= part <= len(parts):
        raise SystemExit(f"'{law}'의 파트는 1~{len(parts)}입니다.")

    articles = parts[part - 1]
    print(
        f"# {law} — 파트 {part}/{len(parts)} (조문 {len(articles)}개, 테마 {articles[0]['theme']})"
    )
    print(f"# 출력 파일: {output_path(law, part).relative_to(PROJECT_ROOT)}\n")
    for article in articles:
        flag = "  [LOW-VALUE]" if article["article_title"] in LOW_VALUE_TITLES else ""
        print(f"=== {article['id']}{flag}")
        print(f"{article['title']}  ({article['chapter'] or '장 없음'})")
        print(article["text"])
        print()


if __name__ == "__main__":
    app()
