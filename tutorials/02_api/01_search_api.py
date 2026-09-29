"""
2단계-1: 모델을 검색 API로 감싸기 (apps/api, FastAPI)
=====================================================

"모델은 만들었는데, 그래서 어떻게 써요?" — 동료·서비스가 부를 수 있게 HTTP API로 감싼다.

학습 목표:
- API 앱이 시작할 때 모델·인덱스를 한 번 로드하고(lifespan), 요청마다 재사용하는 구조를 본다
- 요청/응답 스키마(pydantic)가 문서(/docs)와 검증을 동시에 만들어 준다는 것을 확인한다
- 헬스체크 / 검색 / 법령 필터 검색 / 답변(LLM) 엔드포인트를 차례로 호출한다
- 처음에는 torch 백엔드로 띄운다 → 3단계에서 ONNX로 바꾸며 무엇이 달라지는지 비교한다

이 파일은 서버를 따로 띄우지 않고 같은 앱을 프로세스 안에서 호출한다(TestClient).
실제 서버는 이렇게 띄운다:
    uv run --package ragkit-api ragkit-api --backend torch      # http://127.0.0.1:8000/docs

사전 준비:
    uv run ragkit index            # data/processed/index/multilingual-e5-small.sqlite

실행:
    uv run python tutorials/02_api/01_search_api.py
"""

import json
import time

from fastapi.testclient import TestClient
from ragkit_api.app import create_app


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def show(resp) -> dict:
    data = resp.json()
    print(f"HTTP {resp.status_code}")
    return data


# ============================================================
# 1. 앱 만들기: 모델·인덱스는 시작할 때 한 번만 로드한다
# ============================================================
section("1. 앱 시작 (torch 백엔드)")
start = time.perf_counter()
app = create_app(open_kwargs={"backend": "torch"})
with TestClient(app) as client:  # with 블록에 들어갈 때 lifespan이 모델·인덱스를 연다
    print(f"모델·인덱스 로딩 {time.perf_counter() - start:.1f}초 (이후 요청은 재사용)")

    # ========================================================
    # 2. 헬스체크: 서버가 살아 있고 무엇을 로드했는지
    # ========================================================
    section("2. GET /api/health")
    print(json.dumps(show(client.get("/api/health")), ensure_ascii=False, indent=2))

    # ========================================================
    # 3. 검색: 질문 → 상위 k개 조문
    # ========================================================
    section("3. POST /api/search")
    body = {"query": "야간에 일하면 수당을 더 받나요?", "k": 3}
    print("요청:", json.dumps(body, ensure_ascii=False))
    t = time.perf_counter()
    data = show(client.post("/api/search", json=body))
    print(f"응답 {1000 * (time.perf_counter() - t):.0f}ms")
    for hit in data["hits"]:
        print(f"  {hit['score']:.3f}  {hit['title']}")

    # ========================================================
    # 4. 법령 필터: 특정 법령 안에서만 찾기
    # ========================================================
    section("4. POST /api/search (laws 필터)")
    body = {"query": "야간에 일하면 수당을 더 받나요?", "k": 3, "laws": ["근로기준법"]}
    for hit in show(client.post("/api/search", json=body))["hits"]:
        print(f"  {hit['score']:.3f}  {hit['title']}")

    # ========================================================
    # 5. 스키마 검증: 잘못된 요청은 코드 없이도 422로 막힌다
    # ========================================================
    section("5. 잘못된 요청 → 422 (pydantic 검증)")
    resp = client.post("/api/search", json={"query": "", "k": 999})
    print(f"HTTP {resp.status_code}: {[e['loc'][-1] + ' ' + e['msg'] for e in resp.json()['detail']]}")
    resp = client.post("/api/search", json={"query": "수당", "laws": ["없는법"]})
    print(f"HTTP {resp.status_code}: {resp.json()['detail']}")

    # ========================================================
    # 6. 답변: 검색한 조문만 근거로 LLM이 답한다 (GEMINI_API_KEY 필요)
    # ========================================================
    section("6. POST /api/answer")
    if not client.get("/api/health").json()["llm_available"]:
        print("GEMINI_API_KEY가 없어 건너뜁니다.")
    else:
        body = {"query": "야간에 일하면 수당을 더 받나요?", "k": 5, "laws": ["근로기준법"]}
        data = show(client.post("/api/answer", json=body))
        print(data["answer"].strip() if not data.get("error") else f"오류: {data['error']}")
        print(f"\n근거 {len(data['hits'])}개, 입력 {data['input_tokens']} 토큰, {data['latency_s']:.1f}초")

print("""
정리:
- 모델 로딩(수 초)은 서버 시작 때 한 번, 요청은 수십 ms
- 스키마 하나로 검증·문서(/docs)·클라이언트 계약이 함께 생긴다
- 다음(02): 답변을 조각조각 흘려보내는 스트리밍(SSE)과 노트북 저장
- 3단계: 같은 API를 ONNX 백엔드로 바꾸면 설치 크기·로딩·메모리가 어떻게 바뀌나?
""")
