"""
실습 5-4 (8교시): 모니터링 — MLflow 3로 실험·모델·서비스를 한곳에서 본다
==========================================================================

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

사전 준비 (실습 4-2의 가지치기+INT8 모델 models/multilingual-e5-small-pruned-int8, Drive에서 받거나):
    uv run ragkit index
    uv run ragkit prune-vocab models/multilingual-e5-small
    uv run ragkit export-onnx models/multilingual-e5-small-pruned
    uv run ragkit quantize models/multilingual-e5-small-pruned
    .env의 GEMINI_API_KEY (없으면 llm 스팬이 오류로 남는다. 있으면 LLM 2회 호출)

실행:
    uv run python lecture/05_product/04_monitoring.py            # 끝나면 MLflow UI를 열어 둔다 (Ctrl+C로 끝)
    uv run python lecture/05_product/04_monitoring.py --check    # 확인만 하고 종료 (bench는 test 질문 100개로, 몇 분)

MLflow 서버는 포트 5050에 띄운다 (macOS는 5000을 AirPlay 수신이 쓴다).

저장 위치: mlruns/ (git 무시). 다시 처음부터 하려면 mlruns/를 지운다.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

if importlib.util.find_spec("mlflow") is None:
    sys.exit(
        "mlflow가 설치되어 있지 않습니다. 먼저 설치하세요:\n  uv sync --extra mlflow"
    )

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "mlruns"
MLFLOW_PORT = 5050  # macOS는 5000을 AirPlay 수신이 쓴다
API_PORT = 8766
TRACKING_URI = f"http://127.0.0.1:{MLFLOW_PORT}"
MODEL_DIR = ROOT / "models" / "multilingual-e5-small-pruned-int8"
REGISTRY_URI = "models:/law-embedder@champion"
TEST = ROOT / "data" / "splits" / "test.jsonl"
CHECK = "--check" in sys.argv
CHECK_QUESTIONS = 100  # --check에서 bench에 쓸 test 질문 수

# 이 스크립트가 띄우는 모든 프로세스(ragkit·ragkit-bench·ragkit-api)가 같은 MLflow 서버에 기록한다
ENV = {
    **os.environ,
    "MLFLOW_TRACKING_URI": TRACKING_URI,
    "PYTHONUNBUFFERED": "1",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)


def wait_http(url: str, proc: subprocess.Popen, seconds: int = 60) -> None:
    """url이 응답할 때까지 기다린다. 그 사이 proc이 죽으면 바로 알린다."""
    for _ in range(seconds):
        if proc.poll() is not None:
            sys.exit(
                f"서버 프로세스({url})가 바로 종료되었습니다 (종료 코드 {proc.returncode}). "
                "포트가 이미 쓰이고 있는지 확인하세요."
            )
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


for need in (MODEL_DIR / "onnx" / "model.onnx", TEST):
    if not need.exists():
        sys.exit(
            f"{need}가 없습니다. Drive에서 받거나 실습 4-2(lecture/04_optimize/02_onnx_quantize_prune.py)의 CLI로 만드세요."
        )
tmp = tempfile.TemporaryDirectory()  # --check용 질문 부분집합

# ============================================================
# 1. MLflow 서버 (추적 저장소 SQLite + 아티팩트 폴더)
# ============================================================
section("1. mlflow server → " + TRACKING_URI)
STORE.mkdir(exist_ok=True)
server_cmd = [
    sys.executable,
    "-m",
    "mlflow",
    "server",
    "--backend-store-uri",
    f"sqlite:///{STORE / 'mlflow.db'}",
    "--artifacts-destination",
    str(STORE / "artifacts"),
    "--host",
    "127.0.0.1",
    "--port",
    str(MLFLOW_PORT),
]
print(
    f"$ mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --artifacts-destination mlruns/artifacts --port {MLFLOW_PORT}"
)
mlflow_server = subprocess.Popen(
    server_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
api_server = None
try:
    wait_http(f"{TRACKING_URI}/health", mlflow_server)
    print(
        "서버 준비 완료. 이제 .env에 MLFLOW_TRACKING_URI만 있으면 ragkit 도구들이 여기에 기록한다."
    )

    # ============================================================
    # 2. 실험 추적: 실습 4-3의 배포 벤치를 다시 돌려 run으로 남긴다
    # ============================================================
    section("2. 실험 추적 — ragkit-bench (부모 run + 변형마다 자식 run)")
    questions = TEST
    if CHECK:
        lines = [line for line in TEST.read_text().splitlines() if line.strip()][
            :CHECK_QUESTIONS
        ]
        questions = Path(tmp.name) / "test_subset.jsonl"
        questions.write_text("\n".join(lines) + "\n")
        print(f"(--check: test 질문 앞 {len(lines)}개만 쓴다)")
    run(
        [
            "ragkit-bench",
            "run",
            "--variants",
            "onnx-int8",
            "--variants",
            "onnx-int8-pruned",
            "--questions",
            str(questions),
            "--n-latency",
            "30",
            "--out-dir",
            "mlruns/bench",
        ]
    )
    print("""
같은 방식으로 기록되는 명령 (코드 변경 없이 환경 변수만):
  ragkit train --config …     설정 파라미터, epoch별 dev R@5 곡선, 학습 시간
  ragkit evaluate <모델>       R@k·MRR·nDCG, 질문 유형별 R@5, 결과 JSON
  ragkit compare <config>      부모 run + 모델별 자식 run, 비교표(comparison.md)""")

    # ============================================================
    # 3. 모델 레지스트리: 벤치에서 고른 모델을 등록하고 champion 별칭
    # ============================================================
    section("3. 모델 레지스트리 — ragkit register")
    run(
        [
            "ragkit",
            "register",
            str(MODEL_DIR.relative_to(ROOT)),
            "--name",
            "law-embedder",
            "--alias",
            "champion",
        ]
    )

    # ============================================================
    # 4. 서비스 트레이싱: 레지스트리 모델로 API를 띄우고 노트북에서 대화
    # ============================================================
    section(f"4. ragkit-api --model {REGISTRY_URI}")
    api_cmd = [
        "ragkit-api",
        "--port",
        str(API_PORT),
        "--model",
        REGISTRY_URI,
        "--db",
        str(STORE / "notebooks.sqlite"),
    ]
    print("$ " + " ".join(api_cmd))
    api_server = subprocess.Popen(
        api_cmd, cwd=ROOT, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    wait_http(f"http://127.0.0.1:{API_PORT}/api/health", api_server, seconds=120)

    notebook = json.loads(
        post("/api/notebooks", {"title": "근로 상담", "laws": ["근로기준법"]})
    )
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
    found = []
    # API 서버는 트레이스를 비동기로 보낸다 → 두 질문이 다 도착할 때까지 잠깐 기다림
    for _ in range(30):
        # 첫 트레이스가 도착해야 실험이 생긴다
        exp = mlflow.get_experiment_by_name("ragkit-service")
        if exp is not None:
            found = mlflow.search_traces(
                locations=[exp.experiment_id],
                filter_string=f"metadata.`mlflow.trace.session` = '{notebook['id']}'",
                order_by=["timestamp_ms ASC"],
                return_type="list",
            )
            if len(found) >= 2:
                break
        time.sleep(1)
    if not found:
        print(
            "30초 안에 트레이스가 도착하지 않았습니다. API 서버에 mlflow(또는 mlflow-tracing)가 설치되어 있는지 확인하세요."
        )
    for item in found:
        trace = mlflow.get_trace(item.info.trace_id)
        if trace is None:
            print(f"\n트레이스 {item.info.trace_id}를 읽지 못했습니다.")
            continue
        spans = {s.span_id: s for s in trace.data.spans}
        root = next((s for s in spans.values() if s.parent_id is None), None)
        if root is None:
            print("\n루트 스팬이 없는 트레이스 (아직 기록 중일 수 있다)")
            continue
        print(
            f"\n[{(root.inputs or {}).get('query', '?')}] {trace.info.execution_duration}ms · {trace.info.state}"
        )
        for span in trace.data.spans:
            depth = 0
            parent = span.parent_id
            while parent in spans:
                depth += 1
                parent = spans[parent].parent_id
            ms = (span.end_time_ns - span.start_time_ns) / 1e6
            extra = ""
            if span.span_type == "RETRIEVER":
                extra = " → " + ", ".join(
                    d["id"].split("_", 2)[-1] for d in span.outputs or []
                )
            if span.span_type == "CHAT_MODEL":
                usage = span.attributes.get("mlflow.chat.tokenUsage") or {}
                extra = f" → 토큰 입력 {usage.get('input_tokens')} · 출력 {usage.get('output_tokens')}"
            print(f"  {'  ' * depth}{span.name} ({span.span_type}) {ms:.0f}ms{extra}")
        if root.status.status_code == "ERROR":
            # 오류 이유는 루트 스팬의 error 속성에 남는다 (키가 없으면 llm 스팬 없이 끝난다)
            reason = (
                (root.attributes or {}).get("error")
                or (root.outputs or {}).get("answer")
                or root.status.description
            )
            print(
                f"  오류: {reason or '내용 없음 (GEMINI_API_KEY가 없으면 llm 스팬 없이 이렇게 끝난다)'}"
            )

    if CHECK:
        print("\n확인 완료 (--check).")
    else:
        section("6. MLflow UI에서 보기")
        print(f"""{TRACKING_URI}
  Experiments › ragkit          deploy_bench의 자식 run 선택 → Compare (R@5 · 지연 · 메모리 · 크기)
  Models › law-embedder         버전 · champion 별칭 · index_key 태그
  Experiments › ragkit-service  Traces: 질문마다 span 트리, Sessions: 노트북별 대화 흐름
  (웹 화면으로 대화하려면: lecture/05_product/03_web_app.py와 같은 서버에 --model {REGISTRY_URI})
끝내려면 Ctrl+C""")
        mlflow_server.wait()
except KeyboardInterrupt:
    pass
finally:
    for proc in (api_server, mlflow_server):
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait()
    tmp.cleanup()
