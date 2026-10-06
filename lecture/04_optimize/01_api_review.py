"""
실습 4-1 (7교시): 검색 API 짧은 복습 (apps/api, FastAPI)
========================================================

최적화에 들어가기 전에, 최적화한 모델을 끼울 자리인 검색 API를 한 번 훑는다.
API 자체는 이미 배웠으므로 핵심만 짧게 본다.

학습 목표:
- 앱이 시작할 때 모델·인덱스를 한 번 로드하고(lifespan) 요청마다 재사용하는 구조를 다시 본다
- health / search / 법령 필터 / 잘못된 요청(422)을 차례로 호출한다
- 답변 스트리밍(SSE)을 한 번 받아 본다: hits(근거 먼저) → delta(답 조각) → done(토큰·지연)
- 노트북 API(법령 묶음 → 대화 → 노트)의 흐름을 요약한다 (웹 화면이 이 API만 부른다)
- 처음에는 torch 백엔드로 띄운다 → 실습 4-3에서 ONNX INT8로 바꿔 무엇이 달라지는지 비교한다

이 파일은 서버를 따로 띄우지 않고 같은 앱을 프로세스 안에서 호출한다(TestClient).
노트북 저장소는 임시 폴더에 만들고 끝나면 지운다(받은 data/app/을 건드리지 않는다).
실제 서버는 이렇게 띄운다:
    uv run --package ragkit-api ragkit-api --backend torch      # http://127.0.0.1:8000/docs

사전 준비:
    uv run ragkit index            # data/processed/index/multilingual-e5-small.sqlite
    .env의 GEMINI_API_KEY (없으면 스트리밍은 hits 뒤 done에 오류가 온다. 있으면 LLM 1회 호출)

실행:
    uv run python lecture/04_optimize/01_api_review.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient
from ragkit_api.app import create_app
from ragkit_api.store import NotebookStore

from ragkit.config import get_settings

QUERY = "야간에 일하면 수당을 더 받나요?"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def read_sse(lines) -> list[tuple[str, dict]]:
    """SSE 줄 스트림을 (event, data) 목록으로 모은다. 조각이 올 때마다 바로 출력한다."""
    events, event = [], None
    for line in lines:
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event:
            data = json.loads(line.split(":", 1)[1])
            events.append((event, data))
            if event == "hits":
                first = data["hits"][0]["title"] if data["hits"] else "-"
                print(f"[hits] 근거 {len(data['hits'])}개 먼저 도착: {first}")
            elif event == "delta":
                print(data["text"], end="", flush=True)
            elif event == "done":
                err = f" 오류: {data['error']}" if data.get("error") else ""
                print(
                    f"\n[done] 입력 {data['input_tokens']} / 출력 {data['output_tokens']} 토큰, "
                    f"{data['latency_s']:.1f}초{err}"
                )
            event = None
    return events


index_dir = get_settings().data_dir / "processed" / "index"
if not any(index_dir.glob("*.sqlite")):
    sys.exit(
        f"인덱스가 없습니다: {index_dir}\n  uv run ragkit index  로 만들거나 Drive에서 받으세요."
    )

with tempfile.TemporaryDirectory() as tmp:
    # ============================================================
    # 1. 앱 시작: 모델·인덱스는 시작할 때 한 번만 로드한다
    # ============================================================
    section("1. 앱 시작 (torch 백엔드)")
    start = time.perf_counter()
    app = create_app(
        store=NotebookStore(Path(tmp) / "notebooks.sqlite"),
        open_kwargs={"backend": "torch"},
    )
    # with 블록에 들어갈 때 lifespan이 모델·인덱스를 연다
    with TestClient(app) as client:
        print(
            f"모델·인덱스 로딩 {time.perf_counter() - start:.1f}초 (이후 요청은 재사용)"
        )

        # ========================================================
        # 2. health · search · 법령 필터
        # ========================================================
        section("2. GET /api/health → POST /api/search")
        health = client.get("/api/health").json()
        print(json.dumps(health, ensure_ascii=False))

        body = {"query": QUERY, "k": 3}
        t = time.perf_counter()
        resp = client.post("/api/search", json=body)
        print(
            f"\n검색 {json.dumps(body, ensure_ascii=False)} → HTTP {resp.status_code}, "
            f"{1000 * (time.perf_counter() - t):.0f}ms"
        )
        for hit in resp.json()["hits"]:
            print(f"  {hit['score']:.3f}  {hit['title']}")

        body = {**body, "laws": ["근로기준법"]}
        print("\n법령 필터 laws=['근로기준법']")
        for hit in client.post("/api/search", json=body).json()["hits"]:
            print(f"  {hit['score']:.3f}  {hit['title']}")

        # ========================================================
        # 3. 스키마 검증: 잘못된 요청은 코드 없이도 422로 막힌다
        # ========================================================
        section("3. 잘못된 요청 → 422 (pydantic 검증)")
        resp = client.post("/api/search", json={"query": "", "k": 999})
        print(
            f"HTTP {resp.status_code}: {[str(e['loc'][-1]) + ' ' + e['msg'] for e in resp.json()['detail']]}"
        )
        resp = client.post("/api/search", json={"query": "수당", "laws": ["없는법"]})
        print(f"HTTP {resp.status_code}: {resp.json()['detail']}")

        # ========================================================
        # 4. 스트리밍 답변 (SSE) 한 번
        # ========================================================
        section("4. POST /api/answer/stream (SSE)")
        if not health["llm_available"]:
            print(
                "GEMINI_API_KEY가 없어 LLM 답변 대신 오류가 옵니다. 이벤트 순서만 보세요."
            )
        body = {
            "query": "월세 계약이 끝나면 보증금은 언제 돌려받나요?",
            "k": 5,
            "laws": ["주택임대차보호법"],
        }
        start = time.perf_counter()
        with client.stream("POST", "/api/answer/stream", json=body) as resp:
            print(
                f"HTTP {resp.status_code}, Content-Type: {resp.headers['content-type']}"
            )
            events = read_sse(resp.iter_lines())
        names = [e for e, _ in events]
        print(
            f"이벤트 순서: {names[:3]} … 총 {len(events)}개, {time.perf_counter() - start:.1f}초"
        )

        # ========================================================
        # 5. 노트북 API 흐름 (상태를 서버 SQLite에 저장)
        # ========================================================
        section("5. 노트북 API 흐름 요약")
        nb = client.post(
            "/api/notebooks",
            json={"title": "첫 자취 준비", "laws": ["주택임대차보호법", "근로기준법"]},
        ).json()
        print(f"POST /api/notebooks → {nb['id'][:8]}… 소스 법령 {nb['laws']}")
        print("""이후 흐름 (웹 화면이 부르는 순서, 여기서는 LLM 호출을 아끼려 생략):
  POST   /api/notebooks/{id}/chat       그 법령들 안에서만 검색해 SSE로 답 (이전 대화 참고)
  GET    /api/notebooks/{id}/messages   대화 기록 (답 + 근거 조문)
  POST   /api/notebooks/{id}/notes      답을 근거 조문과 함께 노트로 저장""")
        resp = client.delete(f"/api/notebooks/{nb['id']}")
        print(
            f"DELETE /api/notebooks/{{id}} → HTTP {resp.status_code} (대화·노트 함께 삭제)"
        )

print("""
정리:
- 모델 로딩(수 초)은 서버 시작 때 한 번, 요청은 수십 ms
- 스키마 하나로 검증·문서(/docs)·클라이언트 계약이 함께 생긴다
- SSE는 응답 하나를 열어 둔 채 event/data 줄을 흘려보낸다. 근거를 먼저 보내 체감 속도를 높인다
- 다음(실습 4-2): 이 API에 끼울 모델을 ONNX · INT8 · 어휘 가지치기로 줄인다
""")
