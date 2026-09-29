"""
4단계-3: 사용자 화면 — NotebookLM형 '법령 노트' 웹앱 (apps/web, React)
======================================================================

학습 목표:
- 2단계 API 하나로 화면까지: 노트북(법령 묶음) → 채팅(인용 [1] 달린 스트리밍 답) →
  인용을 누르면 조문 원문 → 답을 노트로 저장
- 프론트엔드는 API만 호출한다 (apps/web/src/api). 모델·인덱스는 서버에만 있다
- 배포 한 줄: 빌드한 화면(apps/web/dist)을 API 서버가 같은 주소에서 함께 제공한다

스택: Vite + React + TypeScript + Tailwind + shadcn/ui + TanStack Query (pnpm)

사전 준비:
    uv run ragkit index
    Node.js 20+, pnpm (corepack enable)

실행:
    uv run python tutorials/04_product/03_web_app.py            # 빌드 후 서버 실행 → 브라우저로 접속
    uv run python tutorials/04_product/03_web_app.py --check    # 화면·API 응답만 확인하고 종료

개발 모드 (화면 코드를 고치며 바로 보기):
    uv run --package ragkit-api ragkit-api            # 터미널 1: API :8000
    cd apps/web && pnpm dev                           # 터미널 2: http://localhost:5173 (/api는 8000으로 프록시)
"""

import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "apps" / "web"
PORT = 8765


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get(path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5) as resp:
        return resp.status, resp.read().decode()


# ============================================================
# 1. 화면 빌드 (정적 파일: html·js·css)
# ============================================================
section("1. pnpm build → apps/web/dist")
if not shutil.which("pnpm"):
    sys.exit("pnpm이 없습니다: corepack enable (Node.js 20+)")
if "--rebuild" in sys.argv or not (WEB / "dist" / "index.html").exists():
    subprocess.run(["pnpm", "install", "--frozen-lockfile"], cwd=WEB, check=True)
    subprocess.run(["pnpm", "build"], cwd=WEB, check=True)
files = [p for p in (WEB / "dist").rglob("*") if p.is_file()]
print(f"dist: 파일 {len(files)}개, {sum(p.stat().st_size for p in files) / 1e3:.0f} KB (이것이 사용자 브라우저로 가는 전부)")

# ============================================================
# 2. API 서버가 화면도 함께 제공
# ============================================================
section("2. ragkit-api --web-dist apps/web/dist")
cmd = ["ragkit-api", "--port", str(PORT), "--web-dist", str(WEB / "dist")]
print("$ " + " ".join(cmd))
server = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            status, body = get("/api/health")
            break
        except OSError:
            time.sleep(1)
    else:
        sys.exit("서버가 60초 안에 뜨지 않았습니다.")
    print(f"GET /api/health → {status} {body}")
    status, html = get("/")
    print(f"GET /           → {status} (화면 index.html, {'<div id=\"root\">' in html and 'React 루트 있음'})")
    status, _ = get("/notebooks/any-id")
    print(f"GET /notebooks/… → {status} (화면 경로도 index.html로 → React Router가 처리)")

    if "--check" in sys.argv:
        print("\n확인 완료 (--check).")
    else:
        section("3. 브라우저로 접속")
        print(f"""http://127.0.0.1:{PORT}
  1) 새 노트북 → 소스 추가(예: 주택임대차보호법, 근로기준법)
  2) "전세 보증금을 지키려면?" → 답의 [1]을 눌러 조문 원문 보기
  3) 답을 노트에 저장, 이어서 "확정일자는 어디서 받아요?" (이전 대화를 참고)
끝내려면 Ctrl+C""")
        server.wait()
except KeyboardInterrupt:
    pass
finally:
    server.terminate()
    server.wait()
