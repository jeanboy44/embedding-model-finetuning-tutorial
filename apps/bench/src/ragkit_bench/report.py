"""비교표 (markdown)."""

COLUMNS = [
    ("variant", "변형", "{}"),
    ("backend", "백엔드", "{}"),
    ("model_mb", "모델 MB", "{:.0f}"),
    ("install_mb", "설치 MB", "{:.0f}"),
    ("load_s", "로딩 s", "{:.2f}"),
    ("latency_ms_p50", "지연 p50 ms", "{:.1f}"),
    ("latency_ms_p95", "지연 p95 ms", "{:.1f}"),
    ("docs_per_s", "문서/초", "{:.0f}"),
    ("peak_rss_mb", "최대 메모리 MB", "{:.0f}"),
    ("recall@1", "R@1", "{:.3f}"),
    ("recall@5", "R@5", "{:.3f}"),
    ("recall@10", "R@10", "{:.3f}"),
    ("mrr@10", "MRR@10", "{:.3f}"),
]


def to_markdown(rows: list[dict]) -> str:
    """변형별 한 줄 비교표. 없는 값은 '-'."""
    lines = [
        "| " + " | ".join(title for _, title, _ in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for row in rows:
        cells = [fmt.format(row[key]) if row.get(key) is not None else "-" for key, _, fmt in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
