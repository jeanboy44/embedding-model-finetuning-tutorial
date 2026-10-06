"""
4단계-1: 동료가 바로 쓰는 검색 CLI (apps/search-cli, ragkit-search)
===================================================================

학습 목표:
- 서버 없이 터미널에서 법령을 찾는 도구를 써 본다: laws → search(법령 필터) → show → ask
- 사람이 읽는 출력과 --json(다른 프로그램이 읽는 출력)을 함께 제공하는 설계를 본다
- 설치 없이 실행하는 배포: uvx. 휠(wheel)을 만들어 torch 없이 돈다는 것을 확인한다

사전 준비:
    uv run ragkit index
    .env에 GEMINI_API_KEY (ask만 필요)

실행:
    uv run python lecture/04_product/01_search_cli.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run(*args: str, show: bool = True) -> str:
    """ragkit-search 명령을 실행하고 출력을 보여 준다."""
    print(f"$ ragkit-search {' '.join(args)}")
    out = subprocess.run(["ragkit-search", *args], capture_output=True, text=True, cwd=ROOT, check=False).stdout
    if show:
        print(out.rstrip())
    return out


# ============================================================
# 1. 사람이 쓰는 명령들
# ============================================================
section("1. 법령 목록 → 검색 → 조문 보기")
run("laws", "--theme", "youth")
print()
run("search", "야간 근로 수당", "-k", "3", "--law", "근로기준법")
print()
run("show", "근로기준법_법률_제56조", "--article")

section("2. 답변 (찾은 조문만 근거, 받는 대로 출력)")
run("ask", "수습 기간에도 최저임금을 다 줘야 하나요?", "--law", "최저임금법")

# ============================================================
# 2. 다른 프로그램이 쓰는 출력
# ============================================================
section("3. --json: 스크립트·다른 도구와 연결")
hits = json.loads(run("search", "전세 보증금", "-k", "2", "--law", "주택임대차보호법", "--json", show=False))
print(f"JSON 결과 {len(hits)}건, 필드: {sorted(hits[0])}")

# ============================================================
# 3. 배포: 설치 없이 실행 (uvx)
# ============================================================
section("4. uvx로 배포 (torch 없는 가벼운 도구)")
dist = Path(tempfile.mkdtemp())
for package in ("ragkit", "ragkit-search"):
    subprocess.run(["uv", "build", "-q", "--package", package, "--wheel", "-o", str(dist)], cwd=ROOT, check=True)
for wheel in sorted(dist.glob("*.whl")):
    print(f"  {wheel.name}  {wheel.stat().st_size / 1e3:.0f} KB")
check = subprocess.run(
    ["uvx", "--isolated", "--with", str(next(dist.glob("ragkit-*.whl"))),
     "--from", str(next(dist.glob("ragkit_search-*.whl"))),
     "python", "-c", "import importlib.util as u; print(u.find_spec('torch') is None)"],
    capture_output=True, text=True, check=False,
)
print(f"uvx 격리 환경에 torch 없음: {check.stdout.strip() == 'True'}")
print(f"""
동료에게 줄 명령 (저장소 경로만 바꾸면 된다):
  export RAGKIT_PROJECT_ROOT={ROOT}
  uvx --from {ROOT}/apps/search-cli ragkit-search search "야간 근로 수당"
또는 휠 두 개를 건네고:
  uvx --with ragkit-0.1.0-py3-none-any.whl --from ragkit_search-0.1.0-py3-none-any.whl ragkit-search laws
""")
