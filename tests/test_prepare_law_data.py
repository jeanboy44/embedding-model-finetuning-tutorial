"""법령 코퍼스 준비 스크립트(scripts/prepare_law_data.py)의 파싱 테스트."""

import importlib.util
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


law = _load_script("prepare_law_data")


def _write_law(path: Path, title: str, date: str, body: str) -> Path:
    path.write_text(
        f"---\n제목: {title}\n공포일자: {date}\n시행일자: {date}\n"
        f"출처: https://www.law.go.kr/법령/{title}\n---\n\n# {title}\n\n{body}",
        encoding="utf-8",
    )
    return path


BODY = """## 제1장 총칙 <개정 2009.5.21>

##### 제1조 (목적)

이 법은 근로조건의 기준을 정한다. <개정 2018.6.12>

##### 제2조

삭제 <2008.3.21>

##### 제2조의2 (정의)

**①** "근로자"란 사람을 말한다.
  1\\. 첫째 <신설 2020.3.31>

## 부칙

##### 제1조 (시행일)

이 법은 공포한 날부터 시행한다.
"""


def test_parse_law_file_splits_articles(tmp_path) -> None:
    """조문을 나누고, 삭제 조문과 부칙은 빼고, 개정 태그와 강조 표시를 지운다."""
    path = _write_law(tmp_path / "법률.md", "테스트법", "2024-01-01", BODY)

    meta, articles = law.parse_law_file(path)

    assert meta["제목"] == "테스트법"
    assert articles == [
        ("제1조", "목적", "제1장 총칙", "이 법은 근로조건의 기준을 정한다."),
        ("제2조의2", "정의", "제1장 총칙", '① "근로자"란 사람을 말한다.\n1. 첫째'),
    ]


def test_latest_law_files_skips_repealed_old_law(tmp_path) -> None:
    """같은 제목이 두 파일에 있으면 공포일자가 최신인 파일만 쓴다."""
    _write_law(
        tmp_path / "법률.md",
        "근로기준법",
        "1997-03-13",
        "##### 제0조\n\n근로기준법은 이를 폐지한다.\n",
    )
    current = _write_law(tmp_path / "법률(법률).md", "근로기준법", "2026-06-09", BODY)
    decree = _write_law(tmp_path / "시행령.md", "근로기준법 시행령", "2025-04-08", BODY)

    assert law.latest_law_files(tmp_path) == [current, decree]


def test_build_corpus_ids_and_fields(tmp_path) -> None:
    """id는 <폴더>_<법령구분>_<조문번호>, category는 폴더 이름이다."""
    law_dir = tmp_path / "kr" / "테스트법"
    law_dir.mkdir(parents=True)
    _write_law(law_dir / "법률(법률).md", "테스트법", "2024-01-01", BODY)

    corpus = law.build_corpus(tmp_path, {"youth": ["테스트법", "없는법"]})

    assert [a.id for a in corpus] == ["테스트법_법률_제1조", "테스트법_법률_제2조의2"]
    assert corpus[0].title == "테스트법 제1조 (목적)"
    assert corpus[0].category == "테스트법"
    assert corpus[0].theme == "youth"


def test_split_article_keeps_short_article_whole() -> None:
    """한도 안의 조문은 나누지 않는다."""
    text = "① 첫째 항이다.\n② 둘째 항이다."

    assert law.split_article(text, heading_chars=10, max_chars=100) == [("", text)]


def test_split_article_by_paragraph() -> None:
    """긴 조문은 항 단위로 나누고, 항 번호를 레이블로 붙인다."""
    text = "① 첫째 항이다.\n② 둘째 항이다.\n⑪ 열한째 항이다."

    assert law.split_article(text, heading_chars=10, max_chars=30) == [
        ("제1항", "① 첫째 항이다."),
        ("제2항", "② 둘째 항이다."),
        ("제11항", "⑪ 열한째 항이다."),
    ]


def test_split_article_by_item_group_repeats_lead() -> None:
    """항이 없거나 항 하나가 여전히 길면 호를 묶어 나누고, 머리 문장을 조각마다 반복한다."""
    lead = "용어의 뜻은 다음과 같다."
    items = [f'{n}. "용어{n}"이란 어떤 것을 말한다.' for n in (1, 2, 3)] + [
        '3의2. "용어3의2"란 어떤 것을 말한다.'
    ]
    text = "\n".join([lead, *items])

    chunks = law.split_article(text, heading_chars=10, max_chars=80)

    assert [label for label, _ in chunks] == ["제1~2호", "제3~3의2호"]
    assert chunks[0][1] == "\n".join([lead, *items[:2]])
    assert chunks[1][1] == "\n".join([lead, *items[2:]])


def test_build_corpus_splits_long_article(tmp_path, monkeypatch) -> None:
    """나눈 조각은 id에 레이블이 붙고 parent_id로 원래 조문을 가리킨다."""
    monkeypatch.setattr(law, "MAX_CHARS", 40)
    law_dir = tmp_path / "kr" / "테스트법"
    law_dir.mkdir(parents=True)
    body = "##### 제3조 (의무)\n\n**①** 사용자는 임금을 지급한다.\n**②** 근로자는 성실히 일한다.\n"
    _write_law(law_dir / "법률(법률).md", "테스트법", "2024-01-01", body)

    corpus = law.build_corpus(tmp_path, {"youth": ["테스트법"]})

    assert [a.id for a in corpus] == ["테스트법_법률_제3조_제1항", "테스트법_법률_제3조_제2항"]
    assert corpus[1].title == "테스트법 제3조 (의무) 제2항"
    assert corpus[1].text == "② 근로자는 성실히 일한다."
    assert corpus[1].parent_id == "테스트법_법률_제3조"
    assert corpus[1].paragraph == "제2항"


def test_parse_law_file_drops_deleted_paragraphs_and_items(tmp_path) -> None:
    """본문 중 삭제된 항·호·목 줄("② 삭제 <2020.1.1>", "3. 삭제", "가. 삭제")은 지운다."""
    body = (
        "##### 제5조 (의무)\n\n"
        "**①** 사용자는 다음을 지킨다.\n"
        "1\\. 첫째\n"
        "2\\. 삭제 <2019.1.1>\n"
        "가\\. 삭제\n"
        "**②** 삭제 <2020.1.1>\n"
        "**③** 근로자는 성실히 일한다.\n"
    )
    path = _write_law(tmp_path / "법률.md", "테스트법", "2024-01-01", body)

    _, articles = law.parse_law_file(path)

    assert articles[0][3] == "① 사용자는 다음을 지킨다.\n1. 첫째\n③ 근로자는 성실히 일한다."


def test_paragraphs_from_16_use_angle_bracket_markers(tmp_path, monkeypatch) -> None:
    """제16항부터는 원문자 대신 `**<16>** <16>` 표기를 쓴다. 이것도 항으로 나누고 표기는 한 번만 남긴다."""
    monkeypatch.setattr(law, "MAX_CHARS", 40)
    law_dir = tmp_path / "kr" / "테스트법"
    law_dir.mkdir(parents=True)
    body = (
        "##### 제7조 (공제)\n\n"
        "**⑮** 열다섯째 항이다.\n"
        "**<16>** <16> 열여섯째 항이다.\n"
        "**<17>** <17> 열일곱째 항이다.\n"
    )
    _write_law(law_dir / "법률(법률).md", "테스트법", "2024-01-01", body)

    corpus = law.build_corpus(tmp_path, {"youth": ["테스트법"]})

    assert [(a.paragraph, a.text) for a in corpus] == [
        ("제15항", "⑮ 열다섯째 항이다."),
        ("제16항", "<16> 열여섯째 항이다."),
        ("제17항", "<17> 열일곱째 항이다."),
    ]


def test_build_corpus_rejects_duplicate_ids(tmp_path, monkeypatch) -> None:
    """조각 id가 겹치면 코퍼스를 쓰기 전에 멈춘다(검색 인덱스는 id가 고유해야 한다)."""
    monkeypatch.setattr(law, "split_article", lambda text, heading_chars: [("제1항", text), ("제1항", text)])
    law_dir = tmp_path / "kr" / "테스트법"
    law_dir.mkdir(parents=True)
    _write_law(law_dir / "법률(법률).md", "테스트법", "2024-01-01", "##### 제1조\n\n① 본문\n")

    import pytest

    with pytest.raises(ValueError, match="중복 id"):
        law.build_corpus(tmp_path, {"youth": ["테스트법"]})
