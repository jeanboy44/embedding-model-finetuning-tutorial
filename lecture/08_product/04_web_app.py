"""
실습 5-4 (8교시): 사용자 화면 — NotebookLM형 '법령 노트' 웹앱 (apps/web, React)
================================================================================

학습 목표:
- 실습 4-1의 API 하나로 화면까지: 노트북(법령 묶음) → 채팅(인용 [1] 달린 스트리밍 답) →
  인용을 누르면 조문 원문 → 답을 노트로 저장
- 프론트엔드는 API만 호출한다 (apps/web/src/api). 모델·인덱스는 서버에만 있다
- 배포 한 줄: 빌드한 화면(apps/web/dist)을 API 서버가 같은 주소에서 함께 제공한다

스택: Vite + React + TypeScript + Tailwind + shadcn/ui + TanStack Query (pnpm)

사전 준비:
    uv run ragkit index             # 또는 Drive에서 받은 data/processed/index/
    Node.js 20+, pnpm (corepack enable)

실행:
    uv run python lecture/08_product/04_web_app.py            # 서버 실행 → 브라우저로 접속 (Ctrl+C로 끝)
    uv run python lecture/08_product/04_web_app.py --check    # 화면·API 응답만 확인하고 바로 종료
    uv run python lecture/08_product/04_web_app.py --run      # 화면을 다시 빌드한 뒤 실행 (pnpm install + build)

플래그:
    기본    apps/web/dist가 있으면 그대로 쓴다 (없을 때만 빌드)
    --run   dist가 있어도 pnpm install · pnpm build로 다시 빌드한다 (화면 코드를 고쳤을 때)
    --check 서버를 열어 두지 않고 health · / · 화면 경로 응답만 확인하고 끝낸다 (--run과 함께 써도 된다)

개발 모드 (화면 코드를 고치며 바로 보기):
    uv run --package ragkit-api ragkit-api            # 터미널 1: API :8000
    cd apps/web && pnpm dev                           # 터미널 2: http://localhost:5173 (/api는 8000으로 프록시)
"""

import shutil
import subprocess
import sys
import tempfile
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
if "--run" in sys.argv or not (WEB / "dist" / "index.html").exists():
    if not shutil.which("pnpm"):
        sys.exit(
            "pnpm이 없습니다: corepack enable (Node.js 20+). 또는 Drive에서 받은 apps/web/dist를 쓰세요."
        )
    subprocess.run(["pnpm", "install", "--frozen-lockfile"], cwd=WEB, check=True)
    subprocess.run(["pnpm", "build"], cwd=WEB, check=True)
files = [p for p in (WEB / "dist").rglob("*") if p.is_file()]
print(
    f"dist: 파일 {len(files)}개, {sum(p.stat().st_size for p in files) / 1e3:.0f} KB (이것이 사용자 브라우저로 가는 전부)"
)

# ============================================================
# 2. API 서버가 화면도 함께 제공
# ============================================================
section("2. ragkit-api --web-dist apps/web/dist")
cmd = ["ragkit-api", "--port", str(PORT), "--web-dist", str(WEB / "dist")]
print("$ " + " ".join(cmd))
# 서버 로그: 바로 죽었을 때 원인을 보여 주려고 모아 둔다. 서버가 사는 동안 열어 두고 finally에서 닫는다
log = tempfile.TemporaryFile(mode="w+")  # noqa: SIM115
server = subprocess.Popen(
    cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
)


def fail(message: str) -> None:
    log.seek(0)
    tail = log.read().strip().splitlines()[-15:]
    print(message)
    print("서버 로그 마지막 줄:\n  " + "\n  ".join(tail or ["(없음)"]))
    sys.exit(1)


try:
    for _ in range(60):
        if server.poll() is not None:
            fail(
                f"서버가 바로 종료되었습니다 (종료 코드 {server.returncode}). 포트 {PORT}를 다른 프로그램이 쓰는지 확인하세요."
            )
        try:
            status, body = get("/api/health")
            break
        except OSError:
            time.sleep(1)
    else:
        fail("서버가 60초 안에 뜨지 않았습니다.")
    print(f"GET /api/health → {status} {body}")
    status, html = get("/")
    has_root = 'id="root"' in html
    print(
        f"GET /           → {status} (화면 index.html, React 루트 {'있음' if has_root else '없음'})"
    )
    status, _ = get("/notebooks/any-id")
    print(
        f"GET /notebooks/… → {status} (화면 경로도 index.html로 → React Router가 처리)"
    )

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
    if server.poll() is None:
        server.terminate()
        server.wait()
    log.close()
