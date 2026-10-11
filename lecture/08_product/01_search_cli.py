"""
실습 4-1 (8교시): 동료가 바로 쓰는 검색 CLI (apps/search-cli, ragkit-search)
============================================================================

학습 목표:
- 서버 없이 터미널에서 법령을 찾는 도구를 써 본다: laws → search(법령 필터) → show
- 사람이 읽는 출력과 --json(다른 프로그램이 읽는 출력)을 함께 제공하는 설계를 본다
- 설치 없이 실행하는 배포: uvx. 휠(wheel)을 만들어 torch 없이 돈다는 것을 확인한다 (--run)

무엇을 하나 (명령은 전부 apps/search-cli의 ragkit-search를 자식 프로세스로 부른다. 안은 ragkit.service.Searcher 하나다):
    1. laws --theme youth → search "야간 근로 수당" -k 3 --law 근로기준법 → show 근로기준법_법률_제56조 --article
       법령 이름 확인 → 법령을 좁혀 검색 → 조 전체 원문. 사람이 읽는 출력이다
    2. search "전세 보증금" -k 2 --law 주택임대차보호법 --json → 결과 건수와 필드 이름만 출력 (다른 프로그램이 읽는 출력)
    3. uvx 배포: 기본 모드는 동료에게 줄 명령만 안내. --run이면 휠 두 개를 빌드해 uvx 격리 환경에 torch가 없는지 확인
    답변(ask) 명령은 이 강의에서 쓰지 않는다. 검색(근거 조문)까지만 본다

두 가지 모드:
- 기본: 저장소에 설치된 ragkit-search로 명령들을 실행해 본다 (수십 초)
- --run: 기본에 더해 휠 두 개를 임시 폴더에 빌드하고, uvx 격리 환경에서 torch 없이 도는지 확인한다
  (uvx가 의존성을 내려받으므로 네트워크가 필요하다. 임시 폴더는 끝나면 지운다)

사전 준비:
    uv run ragkit index             # 또는 Drive에서 받은 data/processed/index/

실행:
    uv run python lecture/08_product/01_search_cli.py
    uv run python lecture/08_product/01_search_cli.py --run
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cyclopts
from cyclopts import Parameter
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console(emoji=False)
err_console = Console(stderr=True, emoji=False)

ROOT = Path(__file__).resolve().parents[2]
ENV = {
    **os.environ,
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}  # mlflow가 설치돼 있으면 나오는 안내 한 줄을 숨긴다


@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(
        False,
        description="휠 두 개를 임시 폴더에 빌드하고, uvx 격리 환경에서 torch 없이 도는지 확인",
    )


def cli(*, options: Options | None = None) -> Options:
    """실습 4-1: 동료가 바로 쓰는 검색 CLI (apps/search-cli, ragkit-search)"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold]{escape(title)}[/]", style="blue")


def run(*args: str, show: bool = True, required: bool = True) -> str:
    """ragkit-search 명령을 실행하고 출력을 보여 준다.

    실패하면 오류 출력(stderr)을 보여 주고, required면 스크립트를 끝낸다.
    """
    console.print(f"[dim]$ ragkit-search {escape(' '.join(args))}[/]", soft_wrap=True)
    proc = subprocess.run(
        ["ragkit-search", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=ENV,
        check=False,
    )
    if show or proc.returncode != 0:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(proc.stderr.rstrip())
        if required:
            err_console.print(
                Panel(
                    f"ragkit-search가 실패했습니다 (종료 코드 {proc.returncode}). 위 오류를 확인하세요.",
                    title="오류",
                    border_style="red",
                )
            )
            sys.exit(1)
        console.print(
            f"[yellow](종료 코드 {proc.returncode}: 실패해도 다음 단계로 넘어간다)[/]"
        )
    return proc.stdout


# ============================================================
# 1. 사람이 쓰는 명령들
# ============================================================
# 에이전트(SKILL.md)도 같은 순서로 부른다: laws로 정확한 법령 이름을 확인하고,
# --law로 좁혀 search한 뒤, 답에 쓸 조는 show --article로 항·호 전체를 읽는다.
# 출력에서 볼 것: laws의 법령 이름 · 조각 수, search의 [유사도]와 조각 id, show의 조 전체 원문과 출처 URL
section("1. 법령 목록 → 검색 → 조문 보기")
run("laws", "--theme", "youth")  # youth 테마 법령 32개 (법률 · 시행령 · 시행규칙)
console.print()
# 근로기준법으로 좁혀 상위 3개
run("search", "야간 근로 수당", "-k", "3", "--law", "근로기준법")
console.print()
run("show", "근로기준법_법률_제56조", "--article")  # 조각 id → 같은 조의 항·호 전체

# ============================================================
# 2. 다른 프로그램이 쓰는 출력
# ============================================================
# 같은 search에 --json을 붙이면 사람용 출력 대신 JSON 목록이 나온다. 이 스크립트가 그것을
# json.loads로 읽듯이, 에이전트 스킬(SKILL.md)과 다른 도구도 이 출력을 읽는다.
# 출력에서 볼 것: 건수 2와 필드 이름 15개 (id · title · score · text · law_name · law_type · parent_id · source_url …)
section("2. --json: 스크립트·다른 도구와 연결")
hits = json.loads(
    run(
        "search",
        "전세 보증금",
        "-k",
        "2",
        "--law",
        "주택임대차보호법",
        "--json",
        show=False,  # JSON 원문은 길어서 화면에 찍지 않고 파싱만 한다
    )
)
console.print(
    f"JSON 결과 [bold]{len(hits)}[/]건, 필드: {escape(str(sorted(hits[0])) if hits else '-')}"
)

# ============================================================
# 3. 배포: 설치 없이 실행 (uvx)
# ============================================================
# ragkit-search는 onnxruntime만 있으면 돌고 torch가 필요 없다. 그래서 휠(wheel) 두 개
# (ragkit, ragkit-search)만 건네면 동료가 uvx로 설치 없이 쓴다.
# --run: uv build로 휠 두 개를 임시 폴더에 만들고, uvx 격리 환경에서 torch가 없는지 묻는다.
# 출력에서 볼 것: 휠 두 개의 크기(KB), "uvx 격리 환경에 torch 없음: True", 동료에게 줄 명령
section("3. uvx로 배포 (torch 없는 가벼운 도구)")
if not args.run:
    console.print("[dim]휠 빌드와 uvx 격리 환경 확인은 --run에서 한다 (네트워크 필요).[/]")
else:
    with tempfile.TemporaryDirectory() as tmp:  # 휠은 임시 폴더에 두고 끝나면 지운다
        dist = Path(tmp)
        # 워크스페이스의 두 패키지만 (torch 없음)
        for package in ("ragkit", "ragkit-search"):
            subprocess.run(
                ["uv", "build", "-q", "--package", package, "--wheel", "-o", str(dist)],
                cwd=ROOT,
                check=True,
            )
        for wheel in sorted(dist.glob("*.whl")):
            console.print(
                f"  {escape(wheel.name)}  [bold]{wheel.stat().st_size / 1e3:.0f} KB[/]"
            )
        check = subprocess.run(
            [
                "uvx",
                "--isolated",  # 저장소의 .venv와 캐시를 쓰지 않는 새 환경 (의존성은 PyPI에서 받는다)
                "--with",
                str(next(dist.glob("ragkit-*.whl"))),
                "--from",
                str(next(dist.glob("ragkit_search-*.whl"))),
                "python",
                "-c",
                "import importlib.util as u; print(u.find_spec('torch') is None)",  # torch가 없으면 True
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            console.print(
                f"[red]uvx 실행 실패 (종료 코드 {check.returncode}, 네트워크를 확인하세요):[/]"
            )
            console.print(
                "\n".join(check.stderr.strip().splitlines()[-5:]), markup=False
            )
        else:
            console.print(
                f"uvx 격리 환경에 torch 없음: [bold]{check.stdout.strip() == 'True'}[/]"
            )

console.print()
console.print("동료에게 줄 명령 (저장소 경로만 바꾸면 된다):")
console.print(f"[dim]  export RAGKIT_PROJECT_ROOT={escape(str(ROOT))}            # macOS · Linux[/]", soft_wrap=True)
console.print(f'[dim]  $env:RAGKIT_PROJECT_ROOT="{escape(str(ROOT))}"   # Windows PowerShell[/]', soft_wrap=True)
console.print(
    f'[dim]  uvx --from {escape(str(ROOT))}/apps/search-cli ragkit-search search "야간 근로 수당"[/]',
    soft_wrap=True,
)
console.print("또는 휠 두 개를 건네고:")
console.print(
    "[dim]  uvx --with ragkit-0.1.0-py3-none-any.whl --from ragkit_search-0.1.0-py3-none-any.whl ragkit-search laws[/]",
    soft_wrap=True,
)
