"""
실습 4-4 (8교시): 웹 화면 — 검색 API 서버가 HTML 한 장을 함께 준다 (apps/web-lite)
=====================================================================================

학습 목표:
- 3교시의 오프라인 파일과 같은 화면을, 모델 · 인덱스를 서버에 두고 웹으로 건넨다 (빌드 없음, HTML 한 장)
- 화면은 API만 부른다 (/api/health · /api/laws · /api/search · /api/articles). 무거운 것은 서버에만 있다
- 배포 한 줄: 같은 ragkit-api 서버가 /api와 화면(/)을 한 주소에서 준다 (--web-dist)
- 모델을 바꾸려면 서버만 다시 띄운다 (--run: 파인튜닝 모델로)

무엇을 하나:
    1. apps/web-lite/index.html이 있는지, 몇 KB인지 본다 (빌드 산출물이 아니라 저장소의 파일 그대로)
    2. ragkit-api --port 8765 --web-dist apps/web-lite 로 서버를 띄우고 /api/health(모델 · 조각 수) · /(화면) 응답을 확인한다
    3. 화면이 부르는 것과 같은 POST /api/search 요청을 보내 상위 3개 조문과 응답 시간을 본다
    4. 브라우저 주소와 해 볼 것을 안내하고 서버를 열어 둔다 (Ctrl+C로 끝). --check면 3에서 끝낸다

사전 준비:
    uv run ragkit index             # 또는 Drive에서 받은 data/processed/index/ (01_doctor.py --fix)
    --run이면 models/finetuned/002_hard_negatives와 그 인덱스 (3교시에 받음)

실행:
    uv run python lecture/08_product/04_web_app.py            # 학습 전 e5(ONNX)로 서버 실행 → 브라우저 (Ctrl+C로 끝)
    uv run python lecture/08_product/04_web_app.py --check    # 응답만 확인하고 바로 종료
    uv run python lecture/08_product/04_web_app.py --run      # 파인튜닝 모델(002_hard_negatives, torch)로 띄운다

직접 치면:
    uv run --package ragkit-api ragkit-api --web-dist apps/web-lite                 # http://127.0.0.1:8000
"""

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cyclopts
from cyclopts import Parameter
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "apps" / "web-lite"
FINETUNED = ROOT / "models" / "finetuned" / "002_hard_negatives"
PORT = 8765  # ragkit-api 기본 포트 8000과 겹치지 않게
QUERY = "편의점 알바 3개월 했는데 주휴수당 받을 수 있나요?"  # 3교시 질문 넷 중 하나 (정답 근로기준법 제55조)


@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(
        False, description="파인튜닝 모델(002_hard_negatives, torch)로 서버를 띄운다"
    )
    check: bool = Field(
        False,
        description="서버를 띄워 응답만 확인하고 바로 끝낸다 (브라우저 안내 생략)",
    )


def cli(*, options: Options | None = None) -> Options:
    """실습 4-4: 웹 화면 — 검색 API 서버가 HTML 한 장을 함께 준다 (apps/web-lite)"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)
CHECK = args.check
RUN = args.run


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get(path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5) as resp:
        return resp.status, resp.read().decode()


def post(path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, json.loads(resp.read().decode())


# ============================================================
# 1. 화면 파일: 빌드 없이 HTML 한 장
# ============================================================
# apps/web(React)은 pnpm build가 필요하지만, 이 화면은 저장소의 파일 그대로다.
# 3교시 오프라인 파일(56MB)과 디자인이 같고, 모델 · 조문이 빠져 있어 18KB다.
# 출력에서 볼 것: 파일 크기 KB (사용자 브라우저로 가는 전부)
section("1. apps/web-lite/index.html (빌드 없음)")
index_html = WEB / "index.html"
if not index_html.exists():
    sys.exit(f"화면 파일이 없습니다: {index_html}")
print(
    f"{index_html.relative_to(ROOT)}: {index_html.stat().st_size / 1e3:.0f} KB (이것이 사용자 브라우저로 가는 전부)"
)
print("모델 · 인덱스는 들어 있지 않다. 화면은 같은 주소의 /api/… 만 부른다")

# ============================================================
# 2. API 서버가 화면도 함께 제공
# ============================================================
# ragkit-api는 시작할 때 모델과 인덱스를 한 번 열고(lifespan), --web-dist 폴더를 / 에서 준다.
# 기본은 학습 전 e5(ONNX). --run이면 파인튜닝 모델을 torch로 연다 (인덱스는 자동으로 그 모델 것).
# 출력에서 볼 것: /api/health의 모델 키(multilingual-e5-small)와 조문 조각 수 25,967개, /가 200으로 검색 폼을 준다는 것
cmd = ["ragkit-api", "--port", str(PORT), "--web-dist", str(WEB)]
if RUN:
    if not (FINETUNED / "model.safetensors").exists():
        sys.exit(
            f"파인튜닝 모델이 없습니다: {FINETUNED}\n"
            "uv run python scripts/finetuned_drive.py download 002_hard_negatives"
        )
    cmd += ["--checkpoint", str(FINETUNED), "--backend", "torch"]
section(f"2. {' '.join(cmd)}")
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
    # 모델 로드가 끝나면 /api/health가 응답한다 (학습 전 e5 ONNX는 수 초, torch는 수십 초)
    for _ in range(90):
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
        fail("서버가 90초 안에 뜨지 않았습니다.")
    health = json.loads(body)
    print(
        f"GET /api/health → {status} 모델 {health['model_key']} · 조문 조각 {health['doc_count']:,}개"
    )
    status, html = get("/")
    has_form = 'id="form"' in html
    print(
        f"GET /           → {status} (화면 index.html, 검색 폼 {'있음' if has_form else '없음'})"
    )

    # ============================================================
    # 3. 화면이 부르는 것과 같은 요청을 직접 보낸다
    # ============================================================
    # 브라우저의 검색 버튼이 보내는 요청과 같다: POST /api/search {query, k, laws}.
    # 서버가 질문을 임베딩하고 인덱스에서 가까운 조각을 찾는다 (Searcher.search). 화면은 결과 JSON을 카드로 그릴 뿐이다.
    # 출력에서 볼 것: HTTP 200, 응답 시간(ms), 상위 3개의 제목과 유사도. 학습 전 e5라 정답(제55조)이 상위에 없을 수 있다
    section("3. POST /api/search (화면의 검색 버튼이 보내는 요청)")
    start = time.perf_counter()
    # laws=None: 법령 필터 없음 (화면에서 법령을 고르면 이름 목록이 들어간다)
    status, result = post("/api/search", {"query": QUERY, "k": 3, "laws": None})
    elapsed = (time.perf_counter() - start) * 1000
    print(f"질문: {QUERY}")
    print(f"HTTP {status}, {elapsed:.0f}ms, 결과 {len(result['hits'])}건")
    for i, hit in enumerate(result["hits"], 1):
        print(f"  {i}. {hit['title']}  (유사도 {hit['score']:.3f})")

    if CHECK:
        print("\n확인 완료 (--check).")
    else:
        # ============================================================
        # 4. 브라우저로 접속: 서버를 열어 두고 해 볼 것을 안내한다
        # ============================================================
        # 화면의 모든 동작은 같은 서버의 /api/… 호출이다. 개발자 도구 Network 탭에서 그 요청을 본다
        section("4. 브라우저로 접속")
        print(f"""http://127.0.0.1:{PORT}
  1) 예시 질문 칩을 눌러 본다 → 3교시 오프라인 파일과 같은 질문, 같은 화면
  2) 테마 · 법령 필터를 걸어 다시 검색한다 (laws 필터가 /api/search로 간다)
  3) 카드의 "같은 조 전체"로 조 전체를 본다 (/api/articles/…)
  4) 브라우저 개발자 도구 Network 탭에서 /api/search 요청과 응답 JSON을 본다
  5) http://127.0.0.1:{PORT}/docs — 같은 서버의 API 설명서
{"파인튜닝 모델(002_hard_negatives)로 떠 있다. 기본 모드와 같은 질문의 순위를 비교해 본다." if RUN else "학습 전 e5로 떠 있다. --run으로 파인튜닝 모델을 띄워 같은 질문을 비교해 본다."}
끝내려면 Ctrl+C""")
        server.wait()
except KeyboardInterrupt:
    pass
finally:
    if server.poll() is None:
        server.terminate()
        server.wait()
    log.close()
