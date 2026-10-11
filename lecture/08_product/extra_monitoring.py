"""
실습 4 선택 (8교시): 모니터링 — MLflow 3로 실험·모델·서비스를 한곳에서 본다
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

무엇을 하나 (ragkit · ragkit-bench · ragkit-api를 자식 프로세스로 부르고, 환경 변수 MLFLOW_TRACKING_URI만 넘긴다):
    1. mlflow server를 포트 5050에 띄운다 (저장소 mlruns/mlflow.db, 아티팩트 mlruns/artifacts)
    2. ragkit-bench run으로 onnx-int8 · onnx-int8-pruned 두 변형을 test 질문으로 재서 부모 run + 자식 run으로 남긴다
       (--check면 test 앞 100개 질문만, 약 90초)
    3. ragkit register로 가지치기 + INT8 모델을 law-embedder로 등록하고 별칭 champion을 붙인다
    4. ragkit-api --model models:/law-embedder@champion 으로 API를 띄우고, 노트북 하나에 질문 두 개를 보낸다
    5. mlflow.search_traces로 그 노트북(세션)의 트레이스를 읽어 스팬 트리를 출력한다
    6. MLflow UI에서 볼 곳을 안내하고 서버를 열어 둔다 (Ctrl+C로 끝). --check면 5에서 끝낸다

설치 (서버·DS 도구는 mlflow, 배포하는 앱은 가벼운 mlflow-tracing):
    uv sync --extra mlflow                    # 이 저장소 (ragkit[mlflow])
    uv pip install "ragkit-api[tracing]"      # 배포할 때 (mlflow-tracing, 레지스트리 기능 없음)

사전 준비 (실습 3-2의 가지치기+INT8 모델 models/multilingual-e5-small-pruned-int8, Drive에서 받거나):
    uv run ragkit index
    uv run ragkit prune-vocab models/multilingual-e5-small
    uv run ragkit export-onnx models/multilingual-e5-small-pruned
    uv run ragkit quantize models/multilingual-e5-small-pruned
    LLM 답변은 쓰지 않으므로 트레이스는 llm 스팬 없이 끝나고 상태가 ERROR로 남는다 (search · embed_query 스팬은 그대로 본다)

실행:
    uv run python lecture/08_product/extra_monitoring.py            # 끝나면 MLflow UI를 열어 둔다 (Ctrl+C로 끝)
    uv run python lecture/08_product/extra_monitoring.py --check    # 확인만 하고 종료 (bench는 test 질문 100개로, 약 90초)

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

import cyclopts
from cyclopts import Parameter
from pydantic import BaseModel, Field

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


@Parameter(name="*")
class Options(BaseModel):
    check: bool = Field(
        False, description="확인만 하고 종료 (bench는 test 질문 100개로, 약 90초)"
    )


def cli(*, options: Options | None = None) -> Options:
    """실습 4 선택: 모니터링 — MLflow 3로 실험·모델·서비스를 한곳에서 본다"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)
CHECK = args.check
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
            f"{need}가 없습니다. Drive에서 받거나 실습 3-2(lecture/07_optimize/02_onnx_quantize_prune.py)의 CLI로 만드세요."
        )
tmp = tempfile.TemporaryDirectory()  # --check용 질문 부분집합

# ============================================================
# 1. MLflow 서버 (추적 저장소 SQLite + 아티팩트 폴더)
# ============================================================
# 실험 run · 모델 레지스트리 · 트레이스가 전부 이 서버 하나에 모인다. 저장은 mlruns/ 아래라
# 지우면 처음부터다. /health가 응답할 때까지 기다린 뒤 다음 단계로 간다.
# 출력에서 볼 것: 서버 명령 한 줄과 "서버 준비 완료"
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
    # 2. 실험 추적: 실습 3-3의 배포 벤치를 다시 돌려 run으로 남긴다
    # ============================================================
    # ragkit-bench는 코드 변경 없이 MLFLOW_TRACKING_URI만 있으면 부모 run(벤치 전체)과
    # 변형마다 자식 run(R@5 · 지연 · 메모리 · 크기)을 남긴다. 변형은 7교시의 INT8과 가지치기 + INT8 둘.
    # 출력에서 볼 것: 변형별 R@5와 지연 표. UI의 Experiments › ragkit에서 자식 run을 Compare로 나란히 본다
    section("2. 실험 추적 — ragkit-bench (부모 run + 변형마다 자식 run)")
    questions = TEST
    if CHECK:
        lines = [
            line
            for line in TEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:CHECK_QUESTIONS]
        questions = Path(tmp.name) / "test_subset.jsonl"
        questions.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
            "30",  # 지연 측정에 쓸 질문 수 (평가 질문과 별개)
            "--out-dir",
            "mlruns/bench",  # 결과 표 · JSON은 mlruns/bench에 (MLflow 아티팩트와 같은 폴더 아래)
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
    # ragkit register는 모델 폴더를 MLflow 모델로 올리고(버전이 하나 늘어난다) 별칭을 옮긴다.
    # 서비스는 models:/law-embedder@champion만 알면 되므로, 다음에 더 좋은 모델이 나오면 별칭만 옮긴다.
    # 출력에서 볼 것: 등록된 이름 · 버전 · 별칭. UI의 Models › law-embedder
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
    # ragkit-api가 --model에 레지스트리 주소를 받으면 champion 버전을 내려받아 연다 (모델 로드까지 최대 120초).
    # 노트북 "근로 상담"을 만들고 /chat에 질문 두 개를 보낸다. 노트북 id가 곧 MLflow 세션 id다.
    # LLM 답변은 쓰지 않으므로 chat은 근거 조문까지만 하고 끝난다.
    # 출력에서 볼 것: 노트북 id와 질문 두 줄
    section(f"4. ragkit-api --model {REGISTRY_URI}")
    api_cmd = [
        "ragkit-api",
        "--port",
        str(API_PORT),
        "--model",
        REGISTRY_URI,
        "--db",
        # 노트북 · 대화 저장소도 mlruns/ 아래에 (저장소 기본 DB를 건드리지 않는다)
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
    # mlflow.search_traces로 실험 ragkit-service에서 이 노트북(세션) 것만 고르고, 트레이스마다 스팬을
    # 부모 → 자식 들여쓰기로 출력한다. RETRIEVER 스팬에는 찾은 조문 id를 덧붙인다.
    # 출력에서 볼 것: 질문마다 answer(CHAIN) › search(RETRIEVER) › embed_query(EMBEDDING) 트리와 ms.
    # llm 스팬이 없고 상태가 ERROR인 것은 LLM을 쓰지 않아서다
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
            # 오류 이유는 루트 스팬의 error 속성에 남는다 (LLM을 쓰지 않으면 llm 스팬 없이 끝난다)
            reason = (
                (root.attributes or {}).get("error")
                or (root.outputs or {}).get("answer")
                or root.status.description
            )
            print(
                f"  오류: {reason or '내용 없음 (LLM을 쓰지 않으면 llm 스팬 없이 이렇게 끝난다)'}"
            )

    if CHECK:
        print("\n확인 완료 (--check).")
    else:
        # ============================================================
        # 6. MLflow UI 안내: 서버를 열어 두고 볼 곳을 알려 준다
        # ============================================================
        section("6. MLflow UI에서 보기")
        print(f"""{TRACKING_URI}
  Experiments › ragkit          deploy_bench의 자식 run 선택 → Compare (R@5 · 지연 · 메모리 · 크기)
  Models › law-embedder         버전 · champion 별칭 · index_key 태그
  Experiments › ragkit-service  Traces: 질문마다 span 트리, Sessions: 노트북별 대화 흐름
  (웹 화면으로 대화하려면: lecture/08_product/04_web_app.py와 같은 서버에 --model {REGISTRY_URI})
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
