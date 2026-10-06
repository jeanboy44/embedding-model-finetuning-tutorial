"""에이전트 스킬(SKILL.md): Claude Code·Gemini CLI가 ragkit-search를 도구로 쓰게 하는 안내서.

두 에이전트는 같은 형식(앞머리 name·description + 본문)을 읽는다. 설치는 스킬 폴더에 SKILL.md를 쓰는 것뿐이다.

    Claude Code   프로젝트 .claude/skills/<이름>/SKILL.md   사용자 ~/.claude/skills/<이름>/SKILL.md
    Gemini CLI    프로젝트 .gemini/skills/<이름>/SKILL.md   사용자 ~/.gemini/skills/<이름>/SKILL.md
"""

import shlex
from importlib.resources import files
from pathlib import Path
from typing import Literal

SKILL_NAME = "korean-law-search"
Agent = Literal["claude", "gemini"]
AGENT_DIRS: dict[str, str] = {"claude": ".claude", "gemini": ".gemini"}


def default_command(project_root: Path) -> str:
    """이 저장소의 환경으로 ragkit-search를 실행하는 명령 (어느 폴더에서 불러도 데이터 경로가 맞는다)."""
    return f"uv run --directory {shlex.quote(str(project_root))} --package ragkit-search ragkit-search"


def option_flags(
    *, model: str | None = None, checkpoint: Path | None = None, backend: str | None = None, index: Path | None = None
) -> str:
    """명령 끝에 붙일 모델·인덱스 옵션. 경로는 절대 경로로 바꾼다 (에이전트가 다른 폴더에서 부르므로)."""
    flags: list[str] = []
    if model:
        flags += ["--model", model]
    if checkpoint:
        flags += ["--checkpoint", str(Path(checkpoint).resolve())]
    if backend:
        flags += ["--backend", backend]
    if index:
        flags += ["--index", str(Path(index).resolve())]
    return "".join(f" {shlex.quote(flag)}" for flag in flags)


def render(command: str, options: str = "") -> str:
    """SKILL.md 템플릿에 실행 명령과 옵션을 채운다."""
    template = files("ragkit_search").joinpath("skill/SKILL.md").read_text(encoding="utf-8")
    return template.replace("{command}", command).replace("{options}", options)


def skill_path(agent: Agent, base: Path, name: str = SKILL_NAME) -> Path:
    """base(프로젝트 폴더 또는 홈) 아래 그 에이전트의 스킬 파일 위치."""
    if agent not in AGENT_DIRS:
        raise ValueError(f"알 수 없는 에이전트: {agent} (가능: {', '.join(AGENT_DIRS)})")
    return base / AGENT_DIRS[agent] / "skills" / name / "SKILL.md"


def install(agent: Agent, base: Path, text: str, name: str = SKILL_NAME) -> Path:
    """스킬 파일을 쓰고 경로를 돌려준다. 이미 있으면 덮어쓴다."""
    path = skill_path(agent, base, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
