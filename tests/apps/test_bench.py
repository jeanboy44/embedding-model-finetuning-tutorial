"""배포 최적화 벤치마크(ragkit-bench) 테스트: 측정·설치 크기·비교표."""

from pathlib import Path

from ragkit_bench import DEFAULT_VARIANTS, install_size, measure, to_markdown

from ragkit.retrieval import build_index

from .conftest import DOCS, fake_embed

QUESTIONS = [
    {"query": "휴일 휴일", "positive_id": "제55조", "query_type": "keyword"},
    {"query": "임대차 임차인", "positive_id": "제4조", "query_type": "keyword"},
]


def test_measure_reports_speed_memory_free_metrics_and_recall(tmp_path: Path) -> None:
    """지연(p50/p95), 처리량, 로딩 시간, Recall@k를 한 딕셔너리로 돌려준다."""
    index = build_index(DOCS, fake_embed, tmp_path / "idx.sqlite", model_key="fake")

    calls: list[str] = []

    def peak_rss() -> float:
        calls.append("rss")
        return 123.0

    row = measure(
        fake_embed, index, DOCS, QUESTIONS,
        format_query=lambda q: "query: " + q, load_s=1.5, n_latency=5, n_passages=4,
        peak_rss_fn=peak_rss,
    )

    assert row["load_s"] == 1.5
    assert row["peak_rss_mb"] == 123.0  # 서비스 구간(로딩 + 질문 처리) 직후에 한 번 잰다
    assert calls == ["rss"]
    assert 0 <= row["latency_ms_p50"] <= row["latency_ms_p95"]
    assert row["docs_per_s"] > 0
    assert row["recall@1"] == 1.0
    assert set(row) >= {"recall@5", "recall@10", "mrr@10"}


def test_install_size_counts_dependency_closure() -> None:
    """패키지와 그 의존성의 설치 크기를 합친다 (torch 경로가 onnx 경로보다 훨씬 크다)."""
    onnx = install_size(["onnxruntime", "tokenizers"])
    torch = install_size(["torch", "transformers"])

    assert onnx > 0
    assert torch > onnx * 2


def test_default_variants_cover_original_onnx_and_int8() -> None:
    """기본 비교 대상: 원본(torch), ONNX(fp32), ONNX(INT8), ONNX(INT8 + 어휘 가지치기)."""
    assert [v.name for v in DEFAULT_VARIANTS] == ["torch-fp32", "onnx-fp32", "onnx-int8", "onnx-int8-pruned"]
    assert [v.backend for v in DEFAULT_VARIANTS] == ["torch", "onnx", "onnx", "onnx"]


def test_to_markdown_table() -> None:
    """변형별 한 줄 비교표를 만든다."""
    rows = [{"variant": "onnx-int8", "backend": "onnx", "model_mb": 118.1, "install_mb": 90.0, "load_s": 0.4,
             "latency_ms_p50": 5.0, "latency_ms_p95": 7.0, "docs_per_s": 300.0, "peak_rss_mb": 400.0,
             "recall@1": 0.3, "recall@5": 0.5, "recall@10": 0.6, "mrr@10": 0.4}]

    table = to_markdown(rows)

    assert "| onnx-int8 | onnx | 118 |" in table
    assert "0.500" in table
