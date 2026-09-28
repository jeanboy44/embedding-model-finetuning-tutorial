"""`ragkit index` 명령 테스트 (로컬 ONNX 모델 사용)."""

import json
from pathlib import Path

import pytest

from ragkit.config import get_settings
from ragkit.retrieval import VectorIndex

LOCAL_ONNX = get_settings().models_dir / "multilingual-e5-small" / "onnx" / "model.onnx"


@pytest.mark.skipif(not LOCAL_ONNX.exists(), reason="로컬 ONNX 모델 없음 (ragkit export-onnx 필요)")
def test_index_command_builds_searchable_file(tmp_path: Path) -> None:
    """코퍼스 JSON으로 인덱스 파일을 만들고, 그 파일로 검색할 수 있다."""
    from ragkit.cli.cli_tool import index_command
    from ragkit.embeddings import create_embedding_fn, format_queries

    corpus = tmp_path / "docs.json"
    corpus.write_text(json.dumps([
        {"id": "근로기준법_제55조", "title": "근로기준법 제55조 (휴일)",
         "text": "사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장하여야 한다.",
         "theme": "youth", "law_name": "근로기준법", "law_type": "법률", "effective_date": "2025-01-01"},
        {"id": "주택임대차보호법_제4조", "title": "주택임대차보호법 제4조 (임대차기간 등)",
         "text": "임대차가 끝난 경우에도 임차인이 보증금을 반환받을 때까지는 임대차관계가 존속되는 것으로 본다.",
         "theme": "youth", "law_name": "주택임대차보호법", "law_type": "법률", "effective_date": "2025-01-01"},
    ], ensure_ascii=False))
    out = tmp_path / "idx.sqlite"

    index_command(corpus=corpus, out=out)

    index = VectorIndex.open(out)
    embed = create_embedding_fn("intfloat/multilingual-e5-small", backend="onnx")
    hit = index.search(embed(format_queries(["알바 주휴수당"]))[0], k=1)[0]
    assert len(index) == 2
    assert index.model_key == "multilingual-e5-small"
    assert hit.id == "근로기준법_제55조"
