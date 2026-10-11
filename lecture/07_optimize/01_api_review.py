"""
실습 3-1 (7교시): 검색 API 짧은 복습 (apps/api, FastAPI)
========================================================

최적화에 들어가기 전에, 최적화한 모델을 끼울 자리인 검색 API를 한 번 훑는다.
API 자체는 이미 배웠으므로 핵심만 짧게 본다.

학습 목표:
- 앱이 시작할 때 모델·인덱스를 한 번 로드하고(lifespan) 요청마다 재사용하는 구조를 다시 본다
- health / search / 법령 필터 / 잘못된 요청(422)을 차례로 호출한다
- 답변 스트리밍(SSE)을 한 번 받아 본다: hits(근거 먼저) → delta(답 조각) → done(토큰·지연)
- 노트북 API(법령 묶음 → 대화 → 노트)의 흐름을 요약한다 (웹 화면이 이 API만 부른다)
- 처음에는 torch 백엔드로 띄운다 → 실습 3-3에서 ONNX INT8로 바꿔 무엇이 달라지는지 비교한다

무엇을 하나 (모드 하나뿐, 로딩 포함 10~20초):
    0. 인덱스 파일(data/processed/index/*.sqlite)이 있는지 확인한다
    1. ragkit_api.app.create_app으로 앱을 만들고 TestClient로 연다. 이때 lifespan이
       Searcher.open(backend="torch")로 모델·인덱스를 한 번 로드한다 → 로딩 시간 출력
    2. GET /api/health로 모델 키·문서 수·LLM 가능 여부를 보고, POST /api/search로
       같은 질문을 필터 없이 / laws=["근로기준법"]으로 검색해 상위 3건을 비교한다
    3. 빈 질문·k=999·없는 법령 이름을 보내 422와 오류 메시지를 본다 (pydantic 검증)
    4. POST /api/answer/stream을 열어 SSE 이벤트를 순서대로 받는다.
       LLM을 쓰지 않으므로 hits 뒤에 바로 done(error)이 온다 → 이벤트 순서만 본다
    5. 노트북을 하나 만들고 지워서 노트북 API(법령 묶음 → 대화 → 노트)의 흐름을 요약한다

이 파일은 서버를 따로 띄우지 않고 같은 앱을 프로세스 안에서 호출한다(TestClient).
노트북 저장소는 임시 폴더에 만들고 끝나면 지운다(받은 data/app/을 건드리지 않는다).
실제 서버는 이렇게 띄운다:
    uv run --package ragkit-api ragkit-api --backend torch      # http://127.0.0.1:8000/docs

사전 준비:
    uv run ragkit index            # data/processed/index/multilingual-e5-small.sqlite

실행:
    uv run python lecture/07_optimize/01_api_review.py
"""

import json
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

from fastapi.testclient import TestClient
from ragkit_api.app import create_app
from ragkit_api.store import NotebookStore

from ragkit.config import get_settings

QUERY = "야간에 일하면 수당을 더 받나요?"  # 2번에서 필터 없이 / 법령 필터로 두 번 검색하는 질문


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def read_sse(lines: Iterable[str]) -> list[tuple[str, dict]]:
    """SSE 줄 스트림을 (event, data) 목록으로 모은다. 조각이 올 때마다 바로 출력한다."""
    # SSE는 "event: 이름" 줄 다음에 "data: JSON" 줄이 오고 빈 줄로 끝난다
    events, event = [], None
    for line in lines:
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event:
            data = json.loads(line.split(":", 1)[1])
            events.append((event, data))
            if event == "hits":  # 근거 조문 k개가 답보다 먼저 온다
                first = data["hits"][0]["title"] if data["hits"] else "-"
                print(f"[hits] 근거 {len(data['hits'])}개 먼저 도착: {first}")
            elif event == "delta":  # 답 조각. LLM을 쓰지 않으면 오지 않는다
                print(data["text"], end="", flush=True)
            elif event == "done":  # 마지막 이벤트. LLM이 없으면 error 칸에 이유
                err = f" 오류: {data['error']}" if data.get("error") else ""
                print(
                    f"\n[done] 입력 {data['input_tokens']} / 출력 {data['output_tokens']} 토큰, "
                    f"{data['latency_s']:.1f}초{err}"
                )
            event = None
    return events


# 0. 준비물 확인: 기본 모델의 인덱스(ragkit index 결과)가 하나라도 있어야 앱이 뜬다
index_dir = get_settings().data_dir / "processed" / "index"
if not any(index_dir.glob("*.sqlite")):
    sys.exit(
        f"인덱스가 없습니다: {index_dir}\n  uv run ragkit index  로 만들거나 Drive에서 받으세요."
    )

with tempfile.TemporaryDirectory() as tmp:
    # ============================================================
    # 1. 앱 시작: 모델·인덱스는 시작할 때 한 번만 로드한다
    # ============================================================
    # create_app(ragkit_api.app)이 FastAPI 앱을 만든다. open_kwargs는 그대로
    # Searcher.open에 넘어가 어떤 모델·백엔드로 열지 정한다 (3-3에서 onnx로 바꾼다).
    # 출력: 로딩 몇 초. 처음 실행은 OS 캐시가 비어 10초 넘게 걸릴 수 있다.
    section("1. 앱 시작 (torch 백엔드)")
    start = time.perf_counter()
    app = create_app(
        # 노트북 DB는 임시 폴더에. 기본 백엔드는 onnx인데 3-3과 비교하려고 torch로 연다
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
        # health는 모델 키·문서 수·llm_available을 돌려준다. search는 Searcher.search를
        # 그대로 감싼 것이라 CLI `ragkit search`와 같은 결과다. 필터 없이 검색하면
        # 소득세법 시행령이 위에 오고, laws로 좁히면 근로기준법 제56조가 1위가 된다.
        # 출력: 첫 검색의 ms는 워밍업이라 느리다(수백 ms에서 1초 넘게). 이후 요청은 수십 ms다.
        section("2. GET /api/health → POST /api/search")
        health = client.get("/api/health").json()
        print(json.dumps(health, ensure_ascii=False))

        body = {"query": QUERY, "k": 3}  # k=3: 상위 3건만
        t = time.perf_counter()
        resp = client.post("/api/search", json=body)
        print(
            f"\n검색 {json.dumps(body, ensure_ascii=False)} → HTTP {resp.status_code}, "
            f"{1000 * (time.perf_counter() - t):.0f}ms"
        )
        for hit in resp.json()["hits"]:
            print(f"  {hit['score']:.3f}  {hit['title']}")

        body = {**body, "laws": ["근로기준법"]}  # 같은 질문을 근로기준법 안에서만
        print("\n법령 필터 laws=['근로기준법']")
        for hit in client.post("/api/search", json=body).json()["hits"]:
            print(f"  {hit['score']:.3f}  {hit['title']}")

        # ========================================================
        # 3. 스키마 검증: 잘못된 요청은 코드 없이도 422로 막힌다
        # ========================================================
        # 첫 요청은 pydantic 스키마(SearchRequest: query 1자 이상, k 50 이하)가 막고,
        # 둘째 요청은 앱의 check_laws가 Searcher.resolve_laws로 법령 이름을 찾지 못해 막는다.
        # 출력: HTTP 422와 어느 칸이 왜 틀렸는지 적힌 메시지.
        section("3. 잘못된 요청 → 422 (pydantic 검증)")
        # 빈 질문, k 상한(50) 초과
        resp = client.post("/api/search", json={"query": "", "k": 999})
        print(
            f"HTTP {resp.status_code}: {[str(e['loc'][-1]) + ' ' + e['msg'] for e in resp.json()['detail']]}"
        )
        # 모르는 법령 이름
        resp = client.post("/api/search", json={"query": "수당", "laws": ["없는법"]})
        print(f"HTTP {resp.status_code}: {resp.json()['detail']}")

        # ========================================================
        # 4. 스트리밍 답변 (SSE) 한 번
        # ========================================================
        # Searcher.answer_stream을 감싼 엔드포인트. 응답 하나를 열어 둔 채 event/data 줄을
        # 흘려보낸다. 이 강의는 LLM 답변을 쓰지 않으므로 hits(근거 5건) 다음에 바로
        # done이 오고 error 칸에 이유가 담긴다. delta는 오지 않는다.
        # (health의 llm_available이 true인 환경이면 hits와 done 사이에 delta 조각이 온다)
        # 출력: Content-Type text/event-stream, 이벤트 순서 ['hits', 'done'] 총 2개.
        section("4. POST /api/answer/stream (SSE)")
        if not health["llm_available"]:
            print(
                "LLM 답변은 이 강의에서 쓰지 않으므로 hits 뒤 done에 error가 옵니다. 이벤트 순서만 보세요."
            )
        body = {
            "query": "월세 계약이 끝나면 보증금은 언제 돌려받나요?",
            "k": 5,  # 근거 조문 5건
            "laws": ["주택임대차보호법"],
        }
        start = time.perf_counter()
        # stream으로 열어 응답을 줄 단위로 받는다
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
        # 8교시 웹 화면이 부르는 API다. NotebookStore(임시 SQLite)에 노트북 한 개를 만들고
        # 바로 지운다. 대화(chat)·노트(notes)는 LLM 답변이 필요해 호출하지 않고 순서만 적는다.
        # 출력: 만든 노트북 id 앞 8자와 소스 법령, DELETE 뒤 HTTP 204.
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
- 다음(실습 3-2): 이 API에 끼울 모델을 ONNX · INT8 · 어휘 가지치기로 줄인다
""")
