"""모델 프로필: 모델마다 다른 입력 형식·백엔드·로딩 옵션·라이선스를 한 곳에 둔다.

인덱스 생성, 평가, 학습, RAG가 모두 같은 규칙(format_query / format_doc)을 쓰게 하기 위함이다.
모르는 모델(파인튜닝한 e5 폴더 등)은 e5 형식을 쓴다.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ragkit.config import get_settings
from ragkit.data import doc_text


@dataclass(frozen=True)
class ModelProfile:
    """임베딩 모델 하나의 사용 규칙."""

    name: str
    format_query: Callable[[str], str]
    format_doc: Callable[[dict], str]
    backends: tuple[str, ...]  # 지원 백엔드. 첫 번째가 기본값이 될 수 있다
    dtype: str | None = None  # st 백엔드 로딩 dtype (예: "float32", "bfloat16")
    trust_remote_code: bool = False
    encode_kwargs: dict = field(default_factory=dict)  # st encode에 넘길 인자
    deployable: bool = True  # 배포 앱에 써도 되는 라이선스인가
    note: str = ""


def _e5_query(text: str) -> str:
    return f"{get_settings().query_prefix}{text}"


def _e5_doc(doc: dict) -> str:
    return f"{get_settings().passage_prefix}{doc_text(doc)}"


E5_STYLE = ModelProfile(
    name="e5-style",
    format_query=_e5_query,
    format_doc=_e5_doc,
    backends=("onnx", "torch", "st"),
)

PROFILES: dict[str, ModelProfile] = {
    "multilingual-e5-small": ModelProfile(
        name="intfloat/multilingual-e5-small",
        format_query=_e5_query,
        format_doc=_e5_doc,
        backends=("onnx", "torch", "st"),
        note="MIT. 파인튜닝 대상, 118M, 384차원",
    ),
    "embeddinggemma-300m": ModelProfile(
        name="google/embeddinggemma-300m",
        format_query=lambda q: f"task: search result | query: {q}",
        format_doc=lambda d: f"title: {d.get('title') or 'none'} | text: {d['text']}",
        backends=("st",),  # 풀링 뒤 Dense 레이어가 있어 torch(평균 풀링만) 백엔드로는 틀린 값이 나온다
        dtype="float32",  # float16 미지원
        note="Gemma 라이선스(HF 동의 필요), 300M, 768차원",
    ),
    "jina-embeddings-v5-text-small": ModelProfile(
        name="jinaai/jina-embeddings-v5-text-small",
        format_query=lambda q: f"Query: {q}",
        format_doc=lambda d: f"Document: {doc_text(d)}",
        backends=("st",),
        dtype="bfloat16",
        trust_remote_code=True,
        encode_kwargs={"task": "retrieval"},  # 검색용 어댑터
        deployable=False,
        note="CC BY-NC(비상업) → 비교 전용, 677M, 1024차원",
    ),
}


def get_profile(model_name: str | Path) -> ModelProfile:
    """모델 이름(허브 id 또는 로컬 폴더)으로 프로필을 찾는다. 없으면 e5 형식."""
    return PROFILES.get(Path(str(model_name)).name, E5_STYLE)
