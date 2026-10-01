"""
2단계-2: 스트리밍 답변(SSE)과 상태 저장 (노트북 API)
======================================================

학습 목표:
- LLM 답변을 다 기다리지 않고 조각조각 흘려보내는 SSE(Server-Sent Events)를 직접 받아 본다
  event: hits  → 근거 조문 먼저 (화면에 바로 보여 줄 수 있다)
  event: delta → 답변 조각들
  event: done  → 전체 답, 토큰 수, 지연
- 서버에 상태를 저장하는 API: 노트북(= 법령 묶음) → 그 안에서 대화 → 답을 노트로 저장
  (4단계 web 화면이 이 API만 호출한다)

사전 준비:
    uv run ragkit index
    .env에 GEMINI_API_KEY (없으면 hits 뒤 done에 오류가 온다)

실행:
    uv run python lecture/02_api/02_streaming_and_notebooks.py
"""

import json
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient
from ragkit_api.app import create_app
from ragkit_api.store import NotebookStore


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
                print(f"[hits] 근거 {len(data['hits'])}개 먼저 도착: {data['hits'][0]['title'] if data['hits'] else '-'}")
            elif event == "delta":
                print(data["text"], end="", flush=True)
            elif event == "done":
                err = f" 오류: {data['error']}" if data.get("error") else ""
                print(f"\n[done] 입력 {data['input_tokens']} / 출력 {data['output_tokens']} 토큰, {data['latency_s']:.1f}초{err}")
            event = None
    return events


# 강의용 임시 저장소 (실제 서버 기본값은 data/app/notebooks.sqlite)
db = Path(tempfile.mkdtemp()) / "notebooks.sqlite"
app = create_app(store=NotebookStore(db))

with TestClient(app) as client:
    # ========================================================
    # 1. 스트리밍 답변
    # ========================================================
    section("1. POST /api/answer/stream (SSE)")
    body = {"query": "월세 계약 기간이 끝나면 보증금은 언제 돌려받나요?", "k": 5, "laws": ["주택임대차보호법"]}
    start = time.perf_counter()
    with client.stream("POST", "/api/answer/stream", json=body) as resp:
        print(f"HTTP {resp.status_code}, Content-Type: {resp.headers['content-type']}")
        events = read_sse(resp.iter_lines())
    print(f"이벤트 순서: {[e for e, _ in events][:3]} … 총 {len(events)}개, {time.perf_counter() - start:.1f}초")

    # ========================================================
    # 2. 노트북: 법령 묶음을 만들고 그 안에서만 대화한다
    # ========================================================
    section("2. 노트북 만들기 → 대화 → 노트 저장")
    nb = client.post("/api/notebooks", json={"title": "첫 자취 준비", "laws": ["주택임대차보호법", "근로기준법"]}).json()
    print(f"노트북 {nb['id'][:8]}… 소스 법령 {nb['laws']}")

    for query in ["전세 보증금을 지키려면 뭘 해야 하나요?", "그럼 확정일자는 어디서 받아요?"]:
        print(f"\n질문: {query}")
        with client.stream("POST", f"/api/notebooks/{nb['id']}/chat", json={"query": query, "k": 5}) as resp:
            events = read_sse(resp.iter_lines())

    messages = client.get(f"/api/notebooks/{nb['id']}/messages").json()
    print(f"\n저장된 대화 {len(messages)}건 (두 번째 질문은 이전 대화를 참고해 답한다)")

    last = messages[-1]
    note = client.post(
        f"/api/notebooks/{nb['id']}/notes",
        json={"title": "확정일자", "content": last["content"], "citations": last["citations"]},
    ).json()
    print(f"노트 저장: {note['title']} (근거 조문 {len(note['citations'])}개 함께 저장)")

    # ========================================================
    # 3. 정리
    # ========================================================
    client.delete(f"/api/notebooks/{nb['id']}")
    print("\n노트북 삭제 (대화·노트 함께 삭제)")

print("""
정리:
- SSE는 HTTP 응답 하나를 열어 둔 채 'event/data' 줄을 흘려보낸다. 근거를 먼저 보내 체감 속도를 높인다
- 상태(노트북·대화·노트)는 서버 SQLite에 저장한다. 답과 함께 근거 조문도 저장해
  인덱스를 다시 만들어도 기록이 깨지지 않는다
- 4단계 web(apps/web)은 이 API만 호출한다: uv run --package ragkit-api ragkit-api --web-dist apps/web/dist
""")
