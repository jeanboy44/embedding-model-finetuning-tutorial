"""어휘 가지치기 테스트: 남긴 어휘로 덮이는 문장은 원본과 같은 벡터, 크기는 줄어든다."""

import json
from pathlib import Path

import pytest

from ragkit.config import get_settings

LOCAL = get_settings().models_dir / "multilingual-e5-small"
pytestmark = pytest.mark.skipif(not (LOCAL / "model.safetensors").exists(), reason="로컬 모델 없음")

CORPUS = [
    "passage: 근로기준법 제55조 (휴일)\n사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장하여야 한다.",
    "passage: 여신전문금융업법 제70조 (벌칙)\n징역형과 벌금형은 병과(倂科)할 수 있다. 국제표준화기구(ISO)",
]


def test_keep_ids_cover_corpus_prefix_hangul_short_ascii_and_specials() -> None:
    """남길 어휘: 코퍼스 토큰 + 질문 앞 문구 + 한글 전체 + 짧은 ASCII + 특수 토큰."""
    from tokenizers import Tokenizer

    from ragkit.models.vocab_prune import vocab_keep_ids

    tok = Tokenizer.from_file(str(LOCAL / "tokenizer.json"))
    keep = vocab_keep_ids(tok, CORPUS, prefixes=["query: "], ascii_max_len=3)
    vocab = tok.get_vocab()

    assert {vocab[t] for t in ("<s>", "<pad>", "</s>", "<unk>", "<mask>")} <= keep
    assert set(tok.encode(CORPUS[1]).ids) <= keep  # 코퍼스의 한자·로마자 포함
    assert set(tok.encode("query: ").ids) <= keep
    assert vocab["▁주휴"] in keep if "▁주휴" in vocab else True
    assert 10_000 < len(keep) < 40_000


def test_pruned_model_gives_identical_embeddings_for_covered_text(tmp_path: Path) -> None:
    """덮이는 문장은 토큰도 벡터도 원본과 같고, 어휘·가중치는 줄어든다."""
    from tokenizers import Tokenizer

    from ragkit.embeddings import create_embedding_fn
    from ragkit.models.vocab_prune import prune_vocab, vocab_keep_ids

    tok = Tokenizer.from_file(str(LOCAL / "tokenizer.json"))
    keep = vocab_keep_ids(tok, CORPUS, prefixes=["query: "], ascii_max_len=3)
    out = prune_vocab(LOCAL, tmp_path / "pruned", keep)

    config = json.loads((out / "config.json").read_text())
    assert config["vocab_size"] == len(keep)
    assert (out / "model.safetensors").stat().st_size < (LOCAL / "model.safetensors").stat().st_size * 0.4

    new_tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    texts = [*CORPUS, "query: 편의점 알바 주휴수당 받을 수 있나요?", "query: DB형 DC형 퇴직연금 차이"]
    for text in texts:
        assert new_tok.encode(text).tokens == tok.encode(text).tokens

    original = create_embedding_fn(str(LOCAL), backend="torch", device="cpu")(texts)
    pruned = create_embedding_fn(str(out), backend="torch", device="cpu")(texts)
    assert (original * pruned).sum(axis=1).min() >= 0.9999


def test_uncovered_text_still_encodes(tmp_path: Path) -> None:
    """남기지 않은 문자(일본어·이모지)가 와도 오류 없이 <unk> 등으로 처리된다."""
    from tokenizers import Tokenizer

    from ragkit.models.vocab_prune import prune_vocab, vocab_keep_ids

    tok = Tokenizer.from_file(str(LOCAL / "tokenizer.json"))
    out = prune_vocab(LOCAL, tmp_path / "pruned", vocab_keep_ids(tok, CORPUS, prefixes=["query: "]))
    enc = Tokenizer.from_file(str(out / "tokenizer.json")).encode("query: こんにちは 🙂")

    assert max(enc.ids) < json.loads((out / "config.json").read_text())["vocab_size"]
