"""
4단계-4: 모니터링 — MLflow 3로 실험·모델·서비스를 한곳에서 본다
=================================================================

학습 목표:
- 실험 추적: ragkit train·evaluate·compare·ragkit-bench가 MLFLOW_TRACKING_URI만 있으면 run을 남긴다
  (코드 변경 없음). 여러 run을 UI에서 나란히 비교한다
- 모델 레지스트리: 고른 모델을 law-embedder로 등록하고 별칭 champion을 붙인다.
  서비스는 models:/law-embedder@champion만 알면 된다 → 별칭만 옮기면 모델 교체
- 서비스 트레이싱: 질문 한 번 = 트레이스 하나
      answer (CHAIN)
      ├─ search (RETRIEVER)     찾은 조문 본문 전체
      │  └─ embed_query (EMBEDDING)
      └─ llm (CHAT_MODEL)       프롬프트 · 답 · 토큰 수
  노트북 하나 = 세션 하나 → 대화 흐름 전체를 세션 화면에서 다시 본다

설치 (서버·DS 도구는 mlflow, 배포하는 앱은 가벼운 mlflow-tracing):
    uv sync --extra mlflow                    # 이 저장소 (ragkit[mlflow])
    uv pip install "ragkit-api[tracing]"      # 배포할 때 (mlflow-tracing, 레지스트리 기능 없음)

사전 준비 (3단계, 가지치기+INT8 모델 models/multilingual-e5-small-pruned-int8):
    uv run ragkit index
    uv run ragkit prune-vocab models/multilingual-e5-small
    uv run ragkit export-onnx models/multilingual-e5-small-pruned
    uv run ragkit quantize models/multilingual-e5-small-pruned

실행:
    uv run python lecture/04_product/04_monitoring.py            # 끝나면 MLflow UI를 열어 둔다
    uv run python lecture/04_product/04_monitoring.py --check    # 확인만 하고 종료

저장 위치: mlruns/ (git 무시). 다시 처음부터 하려면 mlruns/를 지운다.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "mlruns"
MLFLOW_PORT = 5050  # macOS는 5000을 AirPlay 수신이 쓴다
API_PORT = 8766
TRACKING_URI = f"http://127.0.0.1:{MLFLOW_PORT}"
MODEL_DIR = ROOT / "models" / "multilingual-e5-small-pruned-int8"
REGISTRY_URI = "models:/law-embedder@champion"

# 이 스크립트가 띄우는 모든 프로세스(ragkit·ragkit-bench·ragkit-api)가 같은 MLflow 서버에 기록한다
ENV = {**os.environ, "MLFLOW_TRACKING_URI": TRACKING_URI, "PYTHONUNBUFFERED": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)


def wait_http(url: str, seconds: int = 60) -> None:
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(1)
    sys.exit(f"{url}가 {seconds}초 안에 뜨지 않았습니다.")


def post(path: str, body: dict) -> bytes:
    req = urllib.request.Request(
        f"http://127.0.0.1:{API_PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


if not (MODEL_DIR / "onnx" / "model.onnx").exists():
    sys.exit(f"{MODEL_DIR}가 없습니다. 3단계 lecture/03_optimize를 먼저 실행하세요.")

# ============================================================
# 1. MLflow 서버 (추적 저장소 SQLite + 아티팩트 폴더)
# ============================================================
section("1. mlflow server → " + TRACKING_URI)
STORE.mkdir(exist_ok=True)
server_cmd = [
    sys.executable, "-m", "mlflow", "server",
    "--backend-store-uri", f"sqlite:///{STORE / 'mlflow.db'}",
    "--artifacts-destination", str(STORE / "artifacts"),
    "--host", "127.0.0.1", "--port", str(MLFLOW_PORT),
]
print(f"$ mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --artifacts-destination mlruns/artifacts --port {MLFLOW_PORT}")
mlflow_server = subprocess.Popen(server_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
api_server = None
try:
    wait_http(f"{TRACKING_URI}/health")
    print("서버 준비 완료. 이제 .env에 MLFLOW_TRACKING_URI만 있으면 ragkit 도구들이 여기에 기록한다.")

    # ============================================================
    # 2. 실험 추적: 3단계 배포 벤치를 다시 돌려 run으로 남긴다
    # ============================================================
    section("2. 실험 추적 — ragkit-bench (부모 run + 변형마다 자식 run)")
    run([
        "ragkit-bench", "run", "--variants", "onnx-int8", "--variants", "onnx-int8-pruned",
        "--n-latency", "30", "--out-dir", "mlruns/bench",
    ])
    print("""
같은 방식으로 기록되는 명령 (코드 변경 없이 환경 변수만):
  ragkit train --config …     설정 파라미터, epoch별 dev R@5 곡선, 학습 시간
  ragkit evaluate <모델>       R@k·MRR·nDCG, 질문 유형별 R@5, 결과 JSON
  ragkit compare <config>      부모 run + 모델별 자식 run, 비교표(comparison.md)""")

    # ============================================================
    # 3. 모델 레지스트리: 벤치에서 고른 모델을 등록하고 champion 별칭
    # ============================================================
    section("3. 모델 레지스트리 — ragkit register")
    run(["ragkit", "register", str(MODEL_DIR.relative_to(ROOT)), "--name", "law-embedder", "--alias", "champion"])

    # ============================================================
    # 4. 서비스 트레이싱: 레지스트리 모델로 API를 띄우고 노트북에서 대화
    # ============================================================
    section(f"4. ragkit-api --model {REGISTRY_URI}")
    api_cmd = ["ragkit-api", "--port", str(API_PORT), "--model", REGISTRY_URI,
               "--db", str(STORE / "notebooks.sqlite")]
    print("$ " + " ".join(api_cmd))
    api_server = subprocess.Popen(api_cmd, cwd=ROOT, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_http(f"http://127.0.0.1:{API_PORT}/api/health", seconds=120)

    notebook = json.loads(post("/api/notebooks", {"title": "근로 상담", "laws": ["근로기준법"]}))
    print(f"노트북 {notebook['id']} (= MLflow 세션)")
    for question in ("연차 휴가는 며칠인가요?", "그럼 1년 미만 근무자는요?"):
        post(f"/api/notebooks/{notebook['id']}/chat", {"query": question, "k": 3})
        print(f"  질문: {question}")

    # ============================================================
    # 5. 트레이스 읽기 (UI에서 보는 것과 같은 내용을 코드로)
    # ============================================================
    section("5. 세션의 트레이스")
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name("ragkit-service")
    for _ in range(30):  # API 서버는 트레이스를 비동기로 보낸다 → 두 질문이 다 도착할 때까지 잠깐 기다림
        found = mlflow.search_traces(
            locations=[exp.experiment_id],
            filter_string=f"metadata.`mlflow.trace.session` = '{notebook['id']}'",
            order_by=["timestamp_ms ASC"],
            return_type="list",
        )
        if len(found) >= 2:
            break
        time.sleep(1)
    for trace in found:
        trace = mlflow.get_trace(trace.info.trace_id)
        spans = {s.span_id: s for s in trace.data.spans}
        root = next(s for s in spans.values() if s.parent_id is None)
        print(f"\n[{root.inputs['query']}] {trace.info.execution_duration}ms · {trace.info.state}")
        for span in trace.data.spans:
            depth = 0
            parent = span.parent_id
            while parent:
                depth += 1
                parent = spans[parent].parent_id
            ms = (span.end_time_ns - span.start_time_ns) / 1e6
            extra = ""
            if span.span_type == "RETRIEVER":
                extra = " → " + ", ".join(d["id"].split("_", 2)[-1] for d in span.outputs or [])
            if span.span_type == "CHAT_MODEL":
                usage = span.attributes.get("mlflow.chat.tokenUsage") or {}
                extra = f" → 토큰 입력 {usage.get('input_tokens')} · 출력 {usage.get('output_tokens')}"
            print(f"  {'  ' * depth}{span.name} ({span.span_type}) {ms:.0f}ms{extra}")
        if root.status.status_code == "ERROR":
            print(f"  오류: {(root.outputs or {}).get('answer') or root.status.description}")

    if "--check" in sys.argv:
        print("\n확인 완료 (--check).")
    else:
        section("6. MLflow UI에서 보기")
        print(f"""{TRACKING_URI}
  Experiments › ragkit          deploy_bench의 자식 run 선택 → Compare (R@5 · 지연 · 메모리 · 크기)
  Models › law-embedder         버전 · champion 별칭 · index_key 태그
  Experiments › ragkit-service  Traces: 질문마다 span 트리, Sessions: 노트북별 대화 흐름
  (웹 화면으로 대화하려면: lecture/04_product/03_web_app.py와 같은 서버에 --model {REGISTRY_URI})
끝내려면 Ctrl+C""")
        mlflow_server.wait()
except KeyboardInterrupt:
    pass
finally:
    for proc in (api_server, mlflow_server):
        if proc:
            proc.terminate()
            proc.wait()
