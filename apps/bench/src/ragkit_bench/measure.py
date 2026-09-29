"""측정 함수: 지연·처리량·Recall(measure), 설치 크기(install_size), 모델 파일 크기."""

import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import numpy as np
from packaging.requirements import Requirement

from ragkit.data import doc_text
from ragkit.evaluation import evaluate_index

EmbedFn = Callable[..., np.ndarray]


def measure(
    embed_fn: EmbedFn,
    index,
    corpus: list[dict],
    questions: list[dict],
    *,
    format_query: Callable[[str], str],
    load_s: float,
    n_latency: int = 100,
    n_passages: int = 256,
    batch_size: int = 32,
    peak_rss_fn: Callable[[], float] | None = None,
) -> dict:
    """한 모델 변형의 속도와 검색 정확도를 잰다.

    - 지연: 질문 하나를 임베딩하고 인덱스에서 top-10을 찾는 시간 (서비스의 요청 한 건). p50/p95.
    - 처리량: 문서 n_passages개를 배치로 임베딩하는 속도 (인덱싱 속도).
    - 정확도: 전체 질문으로 evaluate_index (Recall@1/5/10, MRR@10).

    Args:
        embed_fn: 이 변형의 임베딩 함수.
        index: 이 변형의 임베딩으로 만든 VectorIndex.
        corpus: 코퍼스 (평가 판정용).
        questions: 평가 질문.
        format_query: 모델 프로필의 쿼리 형식.
        load_s: 모델 로딩에 걸린 시간 (호출 쪽에서 잰다).
        n_latency: 지연을 잴 질문 수.
        n_passages: 처리량을 잴 문서 수.
        batch_size: 처리량 측정 배치 크기.
        peak_rss_fn: 최대 메모리(MB)를 돌려주는 함수. 서비스 구간(로딩 + 질문 처리) 직후에
            한 번 불러 peak_rss_mb로 남긴다. 배치 임베딩·평가가 쓰는 메모리는 섞지 않는다.
    """
    queries = [format_query(q["query"]) for q in questions[:n_latency]]
    embed_fn(queries[:1])  # 워밍업
    latencies = []
    for query in queries:
        start = time.perf_counter()
        index.search(embed_fn([query])[0], k=10)
        latencies.append((time.perf_counter() - start) * 1000)
    peak_rss_mb = peak_rss_fn() if peak_rss_fn else None

    step = max(len(corpus) // n_passages, 1)
    passages = [doc_text(d) for d in corpus[::step][:n_passages]]
    start = time.perf_counter()
    embed_fn(passages, batch_size=batch_size)
    docs_per_s = len(passages) / (time.perf_counter() - start)

    metrics = evaluate_index(index, embed_fn, corpus, questions, format_query=format_query)["doc"]
    return {
        "load_s": load_s,
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "docs_per_s": docs_per_s,
        "peak_rss_mb": peak_rss_mb,
        **metrics,
    }


def install_size(dists: list[str] | tuple[str, ...]) -> int:
    """패키지들과 그 의존성(현재 환경에 설치된 것)의 설치 크기를 바이트로 합친다.

    extra 전용 의존성은 뺀다. 여러 패키지가 같은 의존성을 쓰면 한 번만 센다.
    """
    seen: set[str] = set()
    stack = [d.lower().replace("_", "-") for d in dists]
    total = 0
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for file in dist.files or []:
            path = Path(file.locate())
            if path.is_file():
                total += path.stat().st_size
        for spec in dist.requires or []:
            req = Requirement(spec)
            if req.marker and ("extra" in str(req.marker) or not req.marker.evaluate()):
                continue
            stack.append(req.name.lower().replace("_", "-"))
    return total


def model_size(model: str | Path, backend: str) -> int:
    """배포에 필요한 모델 파일 크기 (onnx: onnx/model.onnx, torch: 가중치 파일)."""
    folder = Path(model)
    if backend == "onnx":
        return (folder / "onnx" / "model.onnx").stat().st_size
    return sum(p.stat().st_size for p in folder.glob("*.safetensors")) or sum(
        p.stat().st_size for p in folder.glob("*.bin")
    )
