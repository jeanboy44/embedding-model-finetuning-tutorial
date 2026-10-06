"""
실습 5-1 (8교시): 동료가 바로 쓰는 검색 CLI (apps/search-cli, ragkit-search)
============================================================================

학습 목표:
- 서버 없이 터미널에서 법령을 찾는 도구를 써 본다: laws → search(법령 필터) → show → ask
- 사람이 읽는 출력과 --json(다른 프로그램이 읽는 출력)을 함께 제공하는 설계를 본다
- 설치 없이 실행하는 배포: uvx. 휠(wheel)을 만들어 torch 없이 돈다는 것을 확인한다 (--run)

두 가지 모드:
- 기본: 저장소에 설치된 ragkit-search로 명령들을 실행해 본다 (수십 초)
- --run: 기본에 더해 휠 두 개를 임시 폴더에 빌드하고, uvx 격리 환경에서 torch 없이 도는지 확인한다
  (uvx가 의존성을 내려받으므로 네트워크가 필요하다. 임시 폴더는 끝나면 지운다)

사전 준비:
    uv run ragkit index             # 또는 Drive에서 받은 data/processed/index/
    .env의 GEMINI_API_KEY (ask만 필요, 있으면 LLM 1회 호출)

실행:
    uv run python lecture/05_product/01_search_cli.py
    uv run python lecture/05_product/01_search_cli.py --run
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = {
    **os.environ,
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}  # mlflow가 설치돼 있으면 나오는 안내 한 줄을 숨긴다


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run(*args: str, show: bool = True, required: bool = True) -> str:
    """ragkit-search 명령을 실행하고 출력을 보여 준다.

    실패하면 오류 출력(stderr)을 보여 주고, required면 스크립트를 끝낸다.
    """
    print(f"$ ragkit-search {' '.join(args)}")
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
            sys.exit(
                f"ragkit-search가 실패했습니다 (종료 코드 {proc.returncode}). 위 오류를 확인하세요."
            )
        print(f"(종료 코드 {proc.returncode}: 실패해도 다음 단계로 넘어간다)")
    return proc.stdout


# ============================================================
# 1. 사람이 쓰는 명령들
# ============================================================
section("1. 법령 목록 → 검색 → 조문 보기")
run("laws", "--theme", "youth")
print()
run("search", "야간 근로 수당", "-k", "3", "--law", "근로기준법")
print()
run("show", "근로기준법_법률_제56조", "--article")

# ============================================================
# 2. 답변: 찾은 조문만 근거로, 받는 대로 출력
# ============================================================
section("2. 답변 (찾은 조문만 근거, 받는 대로 출력)")
# GEMINI_API_KEY가 없으면 실패한다 → 오류를 보여 주고 다음으로 넘어간다
run(
    "ask",
    "수습 기간에도 최저임금을 다 줘야 하나요?",
    "--law",
    "최저임금법",
    required=False,
)

# ============================================================
# 3. 다른 프로그램이 쓰는 출력
# ============================================================
section("3. --json: 스크립트·다른 도구와 연결")
hits = json.loads(
    run(
        "search",
        "전세 보증금",
        "-k",
        "2",
        "--law",
        "주택임대차보호법",
        "--json",
        show=False,
    )
)
print(f"JSON 결과 {len(hits)}건, 필드: {sorted(hits[0]) if hits else '-'}")

# ============================================================
# 4. 배포: 설치 없이 실행 (uvx)
# ============================================================
section("4. uvx로 배포 (torch 없는 가벼운 도구)")
if "--run" not in sys.argv:
    print("휠 빌드와 uvx 격리 환경 확인은 --run에서 한다 (네트워크 필요).")
else:
    with tempfile.TemporaryDirectory() as tmp:
        dist = Path(tmp)
        for package in ("ragkit", "ragkit-search"):
            subprocess.run(
                ["uv", "build", "-q", "--package", package, "--wheel", "-o", str(dist)],
                cwd=ROOT,
                check=True,
            )
        for wheel in sorted(dist.glob("*.whl")):
            print(f"  {wheel.name}  {wheel.stat().st_size / 1e3:.0f} KB")
        check = subprocess.run(
            [
                "uvx",
                "--isolated",
                "--with",
                str(next(dist.glob("ragkit-*.whl"))),
                "--from",
                str(next(dist.glob("ragkit_search-*.whl"))),
                "python",
                "-c",
                "import importlib.util as u; print(u.find_spec('torch') is None)",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            print(
                f"uvx 실행 실패 (종료 코드 {check.returncode}, 네트워크를 확인하세요):"
            )
            print("\n".join(check.stderr.strip().splitlines()[-5:]))
        else:
            print(f"uvx 격리 환경에 torch 없음: {check.stdout.strip() == 'True'}")

print(f"""
동료에게 줄 명령 (저장소 경로만 바꾸면 된다):
  export RAGKIT_PROJECT_ROOT={ROOT}
  uvx --from {ROOT}/apps/search-cli ragkit-search search "야간 근로 수당"
또는 휠 두 개를 건네고:
  uvx --with ragkit-0.1.0-py3-none-any.whl --from ragkit_search-0.1.0-py3-none-any.whl ragkit-search laws
""")
