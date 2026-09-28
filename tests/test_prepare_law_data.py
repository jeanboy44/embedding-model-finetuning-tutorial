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
