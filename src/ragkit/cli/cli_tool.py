"""ragkit CLI — DS용 도구 (1·3단계): 인덱스·분할·학습·평가·비교·ONNX 변환·양자화.

최종 사용자용 검색은 apps/search-cli(ragkit-search), 서비스는 apps/api가 맡는다.
"""

from pathlib import Path

import cyclopts

from ragkit.cli import train_cli
from ragkit.config import get_settings
from ragkit.embeddings import create_embedding_fn

app = cyclopts.App(name="ragkit", help="ragkit: 임베딩 모델 인덱싱·학습·평가·최적화 (DS용)")

# DS용 학습·평가 명령
for _command in (
    train_cli.split,
    train_cli.train,
    train_cli.evaluate,
    train_cli.compare,
    train_cli.expand,
):
    app.command(_command)


@app.command(name="index")
def index_command(
    corpus: Path | None = None,
    model: str | None = None,
    checkpoint: Path | None = None,
    backend: str | None = None,
    out: Path | None = None,
    device: str | None = None,
    batch_size: int = 64,
    sort_by_length: bool | None = None,
    threads: int | None = None,
) -> None:
    """코퍼스를 임베딩해 SQLite(sqlite-vec) 인덱스 파일을 만든다.

    같은 모델·같은 코퍼스의 파일이 이미 있으면 다시 만들지 않는다.

    Args:
        corpus: 코퍼스 JSON. 기본값 data/processed/law_docs.json.
        model: 임베딩 모델 이름. 기본값 Settings.embedding_model_name.
        checkpoint: 파인튜닝한 모델 폴더. 주면 이 폴더 이름이 모델 키가 된다.
        backend: onnx | torch. 기본값 Settings.embedding_backend.
        out: 인덱스 파일. 기본값 data/processed/index/<모델 키>.sqlite.
        device: torch 장치 auto | cuda | mps | cpu. 기본값 Settings.embedding_device.
        batch_size: 임베딩 배치 크기.
        sort_by_length: 길이순 배치 (--no-sort-by-length로 끔). 기본값 Settings 값.
        threads: CPU 스레드 수. 기본값 라이브러리 기본값.
    """
    from ragkit.data import load_corpus
    from ragkit.embeddings import get_profile
    from ragkit.retrieval import build_index, default_index_path, model_key

    settings = get_settings()
    corpus = corpus or settings.data_dir / "processed" / "law_docs.json"
    model = model or settings.embedding_model_name
    key = model_key(model, checkpoint)
    out = out or default_index_path(key)
    profile = get_profile(checkpoint or model)

    docs = load_corpus(corpus)
    embed_fn = create_embedding_fn(
        model,
        checkpoint_path=checkpoint,
        device=device,
        backend=backend,
        sort_by_length=sort_by_length,
        num_threads=threads,
    )
    index = build_index(
        docs, embed_fn, out, model_key=key, format_doc=profile.format_doc, batch_size=batch_size
    )
    print(f"인덱스: {out} (문서 {len(index):,}개, 모델 {index.model_key})")


@app.command(name="export-onnx")
def export_onnx_command(
    model_dir: Path,
    out_dir: Path | None = None,
) -> None:
    """임베딩 모델 폴더를 ONNX로 변환한다 (extra [train] 필요).

    Args:
        model_dir: config.json과 가중치가 있는 모델 폴더 (base 또는 파인튜닝 결과).
        out_dir: 출력 폴더. 없으면 model_dir/onnx/model.onnx에 쓴다.
    """
    from ragkit.models.onnx_export import export_onnx

    onnx_path = export_onnx(model_dir, out_dir)
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"ONNX 변환 완료: {onnx_path} ({size_mb:.1f} MB)")


@app.command(name="quantize")
def quantize_command(
    model_dir: Path,
    out_dir: Path | None = None,
    per_channel: bool = True,
) -> None:
    """ONNX 모델을 동적 INT8로 양자화한다 (extra [train] 필요).

    Args:
        model_dir: export-onnx를 마친 모델 폴더.
        out_dir: 출력 폴더. 기본값 <model_dir>-int8.
        per_channel: 채널별 양자화 (--no-per-channel이면 텐서 단위, 정확도 비교용).
    """
    from ragkit.models.onnx_export import quantize_onnx

    out_dir = out_dir or model_dir.with_name(f"{model_dir.name}-int8")
    src = model_dir / "onnx" / "model.onnx"
    dst = quantize_onnx(model_dir, out_dir, per_channel=per_channel)
    print(f"INT8 양자화 완료: {dst} ({dst.stat().st_size / 1e6:.1f} MB, 원본 {src.stat().st_size / 1e6:.1f} MB)")
    print(f"사용: create_embedding_fn('{out_dir}', backend='onnx') / uv run ragkit index --model {out_dir}")


@app.command(name="prune-vocab")
def prune_vocab_command(
    model_dir: Path,
    out_dir: Path | None = None,
    corpus: Path | None = None,
    ascii_max_len: int = 3,
    keep_hangul: bool = True,
) -> None:
    """어휘를 코퍼스에 필요한 토큰만 남겨 모델을 줄인다 (extra [torch] 필요).

    남긴 어휘로 토큰화되는 문장은 원본과 같은 벡터가 나온다. 이어서 export-onnx → quantize.

    Args:
        model_dir: HF 모델 폴더 (예: models/multilingual-e5-small, 파인튜닝 결과 폴더).
        out_dir: 출력 폴더. 기본값 <model_dir>-pruned.
        corpus: 코퍼스 JSON. 기본값 data/processed/law_docs.json.
        ascii_max_len: 이 길이 이하의 ASCII 조각을 남긴다 (영문 약어·숫자·기호).
        keep_hangul: 한글 조각을 모두 남긴다 (--no-keep-hangul이면 코퍼스에 나온 것만).
    """
    from tokenizers import Tokenizer

    from ragkit.data import load_corpus
    from ragkit.embeddings import get_profile
    from ragkit.models.vocab_prune import prune_vocab, vocab_keep_ids

    out_dir = out_dir or model_dir.with_name(f"{model_dir.name}-pruned")
    docs = load_corpus(corpus or get_settings().data_dir / "processed" / "law_docs.json")
    profile = get_profile(model_dir)
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    keep = vocab_keep_ids(
        tok,
        (profile.format_doc(d) for d in docs),
        prefixes=[profile.format_query(""), profile.format_doc({"title": "", "text": ""})],
        keep_hangul=keep_hangul,
        ascii_max_len=ascii_max_len,
    )
    prune_vocab(model_dir, out_dir, keep)
    before = (model_dir / "model.safetensors").stat().st_size / 1e6
    after = (out_dir / "model.safetensors").stat().st_size / 1e6
    print(f"어휘 {tok.get_vocab_size():,} → {len(keep):,}개, 가중치 {before:.0f} → {after:.0f} MB: {out_dir}")
    print(f"다음: uv run ragkit export-onnx {out_dir} && uv run ragkit quantize {out_dir}")


if __name__ == "__main__":
    app()
