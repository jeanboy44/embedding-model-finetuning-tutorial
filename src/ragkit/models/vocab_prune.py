"""어휘 가지치기: 다국어 모델의 어휘에서 이 도메인(한국어 법령)에 필요한 토큰만 남긴다 (extra `[torch]`).

e5-small 파라미터의 82%가 단어 임베딩 표(어휘 25만 × 384)다. 법령 코퍼스는 그중 약 2%만 쓴다.
남긴 토큰의 가중치는 그대로이고 트랜스포머 층도 그대로라, 남긴 어휘로만 토큰화되는 문장은
원본과 똑같은 벡터가 나온다(양자화 같은 근사가 아니다). Unigram 토크나이저는 남은 조각 중에서
확률이 가장 높은 분할을 고르므로, 원래 분할이 남긴 조각만 썼다면 분할도 바뀌지 않는다.
달라지는 것은 남기지 않은 조각이 필요한 입력뿐이다(더 잘게 쪼개지거나 <unk>).
"""

import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

HANGUL = re.compile(r"[가-힣]")
ASCII = re.compile(r"[\x00-\x7f▁]+")
SPECIALS = ("<s>", "<pad>", "</s>", "<unk>", "<mask>")
COPY_FILES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
)


def vocab_keep_ids(
    tokenizer,
    texts: Iterable[str],
    *,
    prefixes: Iterable[str] = ("query: ", "passage: "),
    keep_hangul: bool = True,
    ascii_max_len: int = 3,
) -> set[int]:
    """남길 토큰 id: 코퍼스에 나온 토큰 + 앞 문구 + 한글 조각 전체 + 짧은 ASCII 조각 + 특수 토큰.

    Args:
        tokenizer: tokenizers.Tokenizer (원본).
        texts: 코퍼스 문장 (모델 입력 형식 그대로, 예: "passage: 제목\\n본문").
        prefixes: 질문·문서 앞 문구. 코퍼스에 없는 질문 앞 문구도 남기기 위함이다.
        keep_hangul: 한글이 든 조각을 모두 남긴다 (코퍼스에 없는 일상어 질문 대비).
        ascii_max_len: 이 길이 이하의 ASCII 조각을 남긴다 (영문 약어·숫자·기호, 예: DB, 5%).
            한 글자 조각이 남으므로 긴 영어 단어도 <unk>가 아니라 잘게 쪼개진다.
    """
    vocab = tokenizer.get_vocab()
    keep = {vocab[t] for t in SPECIALS if t in vocab}
    for prefix in prefixes:
        keep.update(tokenizer.encode(prefix, add_special_tokens=False).ids)
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == 2000:
            for enc in tokenizer.encode_batch(batch):
                keep.update(enc.ids)
            batch = []
    for enc in tokenizer.encode_batch(batch):
        keep.update(enc.ids)
    for token, token_id in vocab.items():
        is_hangul = keep_hangul and HANGUL.search(token)
        is_short_ascii = ASCII.fullmatch(token) and len(token.lstrip("▁")) <= ascii_max_len
        if is_hangul or is_short_ascii:
            keep.add(token_id)
    return keep


def prune_vocab(model_dir: Path, out_dir: Path, keep_ids: set[int]) -> Path:
    """HF 모델 폴더에서 keep_ids 토큰만 남긴 새 모델 폴더를 만든다.

    토크나이저(tokenizer.json)의 어휘·특수 토큰 id를 새 번호로 바꾸고, 단어 임베딩 표에서
    같은 순서로 행을 골라 저장한다. 결과 폴더는 torch 백엔드로 바로 쓰고,
    `ragkit export-onnx` → `ragkit quantize`로 배포용 ONNX INT8을 만든다.

    Returns:
        출력 폴더.
    """
    try:
        from safetensors.numpy import load_file, save_file
    except ImportError as e:
        raise ImportError("어휘 가지치기에는 extra가 필요합니다: uv sync --extra torch") from e

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(keep_ids)
    new_id = {old: new for new, old in enumerate(order)}

    tok = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    old_vocab = tok["model"]["vocab"]
    tok["model"]["vocab"] = [old_vocab[i] for i in order]
    tok["model"]["unk_id"] = new_id[tok["model"]["unk_id"]]
    for added in tok.get("added_tokens", []):
        added["id"] = new_id[added["id"]]
    for special in (tok.get("post_processor") or {}).get("special_tokens", {}).values():
        special["ids"] = [new_id[i] for i in special["ids"]]
    (out_dir / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")

    weights = load_file(str(model_dir / "model.safetensors"))
    key = next(k for k in weights if k.endswith("word_embeddings.weight"))
    weights[key] = weights[key][order].copy()
    save_file(weights, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config["vocab_size"] = len(order)
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in COPY_FILES:
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, out_dir / name)
    if (model_dir / "1_Pooling").is_dir():
        shutil.copytree(model_dir / "1_Pooling", out_dir / "1_Pooling", dirs_exist_ok=True)
    return out_dir
