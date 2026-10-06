"""apps/search-cli 테스트: 가짜 Searcher를 끼워 명령 출력·종료 코드를 확인한다."""

import json

import pytest
from ragkit_search import cli
from ragkit_search.cli import app


@pytest.fixture
def run(monkeypatch, capsys):
    """토큰 목록으로 명령을 실행하고 (종료 코드, stdout, stderr)를 돌려준다."""

    def _run(tokens: list[str], searcher) -> tuple[int, str, str]:
        monkeypatch.setattr(cli, "_open_searcher", lambda options: searcher)
        code = app(tokens, exit_on_error=False, result_action="return_value")
        out = capsys.readouterr()
        return code, out.out, out.err

    return _run


def test_search_prints_ranked_hits(run, searcher) -> None:
    code, out, _ = run(["search", "휴일", "-k", "2"], searcher)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith(" 1. [") and "근로기준법 제55조 (제55조)" in lines[0]
    assert lines[1].strip() == "휴일 휴일 휴일"
    assert len([line for line in lines if line[:3].strip().endswith(".")]) == 2


def test_search_json_with_law_filter(run, searcher) -> None:
    code, out, _ = run(["search", "임금", "--law", "최저임금법", "--json"], searcher)
    assert code == 0
    hits = json.loads(out)
    assert [h["id"] for h in hits] == ["제6조"]
    assert hits[0]["law_name"] == "최저임금법" and "score" in hits[0]


def test_search_unknown_law_fails(run, searcher) -> None:
    code, out, err = run(["search", "임금", "--law", "임금법"], searcher)
    assert code == 1 and out == ""
    assert err.startswith("오류: 모르는 법령: 임금법") and "최저임금법" in err


def test_missing_index_fails(run, monkeypatch, capsys) -> None:
    def missing(options):
        raise FileNotFoundError("인덱스가 없습니다: x.sqlite")

    monkeypatch.setattr(cli, "_open_searcher", missing)
    code = app(["laws"], exit_on_error=False, result_action="return_value")
    assert code == 1
    assert capsys.readouterr().err.strip() == "오류: 인덱스가 없습니다: x.sqlite"


def test_ask_streams_answer_then_sources(run, searcher) -> None:
    code, out, err = run(["ask", "휴일 근로", "-k", "2"], searcher)
    assert code == 0 and err == ""
    assert out.startswith("휴일에는 가산 임금을 받습니다 [1]\n")
    assert "근거 조문:" in out and "  [1] " in out and "  [2] " in out


def test_ask_without_llm_prints_sources_and_error(run, searcher_without_llm) -> None:
    code, out, err = run(["ask", "휴일"], searcher_without_llm)
    assert code == 1
    assert "근거 조문:" in out and "[1] 근로기준법 제55조 (제55조)" in out
    assert "GEMINI_API_KEY" in err


def test_laws_theme_and_json(run, searcher) -> None:
    code, out, _ = run(["laws", "--theme", "tax"], searcher)
    assert code == 0 and out.strip() == "최저임금법  (법률 · tax · 조각 1개)"
    code, out, _ = run(["laws", "--json"], searcher)
    names = {law["law_name"]: law["doc_count"] for law in json.loads(out)}
    assert names == {"근로기준법": 3, "주택임대차보호법": 1, "최저임금법": 1}


def test_show_piece_and_article(run, searcher) -> None:
    code, out, _ = run(["show", "제56조_제2항"], searcher)
    assert code == 0
    assert out.startswith("근로기준법 제56조_제2항 (제56조_제2항)\n연장 근로\n")
    assert "출처: https://www.law.go.kr/법령/근로기준법" in out

    code, out, _ = run(["show", "제56조_제2항", "--article"], searcher)
    assert "휴일 근로 임금" in out and "연장 근로" in out
    assert out.index("제56조_제1항") < out.index("제56조_제2항")


def test_show_unknown_id_fails(run, searcher) -> None:
    code, _, err = run(["show", "제999조"], searcher)
    assert code == 1 and "없는 조문 id: 제999조" in err


def test_skill_show_fills_command_and_model_options(run, tmp_path) -> None:
    """설치할 SKILL.md에 실행 명령과 모델 옵션이 박힌다 (에이전트가 그 모델로 검색하게)."""
    code, out, _ = run(
        ["skill", "show", "--command", "ragkit-search", "--checkpoint", str(tmp_path / "ft"), "--backend", "torch"], None
    )

    assert code == 0
    assert out.startswith("---\nname: korean-law-search\n")
    assert f'ragkit-search search "<검색어>" -k 5 --json --checkpoint {tmp_path / "ft"} --backend torch' in out
    assert "{command}" not in out and "{options}" not in out


@pytest.mark.parametrize(("agent", "folder"), [("claude", ".claude"), ("gemini", ".gemini")])
def test_skill_install_writes_agent_skill_folder(run, tmp_path, agent, folder) -> None:
    """Claude Code는 .claude/skills/, Gemini CLI는 .gemini/skills/ 아래에 같은 SKILL.md를 쓴다."""
    code, out, _ = run(["skill", "install", "--agent", agent, "--project-dir", str(tmp_path), "--command", "x"], None)

    path = tmp_path / folder / "skills" / "korean-law-search" / "SKILL.md"
    assert code == 0
    assert str(path) in out
    assert 'x search "<검색어>" -k 5 --json\n' in path.read_text(encoding="utf-8")


def test_default_skill_command_runs_from_repo(tmp_path) -> None:
    """기본 실행 명령은 저장소 환경(uv run --directory)이라 에이전트가 어느 폴더에서 불러도 된다."""
    from ragkit_search import agent_skill

    assert agent_skill.default_command(tmp_path) == f"uv run --directory {tmp_path} --package ragkit-search ragkit-search"
