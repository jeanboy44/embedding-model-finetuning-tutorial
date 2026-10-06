"""앱(api · search-cli · mcp)이 쓰는 검색 입구: 모델·인덱스를 한 번 열고 검색·조회·답변을 한다.

    searcher = Searcher.open()                       # Settings 기본 모델 + data/processed/index/<모델>.sqlite
    searcher.search("야간 근로 수당", laws=["근로기준법"])
    for event in searcher.answer_stream("주휴수당은?"):  # HitsEvent → DeltaEvent... → DoneEvent
        ...

모델마다 다른 쿼리 형식(e5의 "query: " 등)은 모델 프로필이 처리하므로 앱은 신경 쓰지 않는다.
"""

import difflib
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ragkit import tracking
from ragkit.config import get_settings
from ragkit.embeddings.profiles import E5_STYLE, ModelProfile, get_profile
from ragkit.models.gemini_client import Generation
from ragkit.rag.answer import AnswerResult, build_prompt
from ragkit.retrieval import (
    LawInfo,
    SearchHit,
    VectorIndex,
    default_index_path,
    model_key,
)

EmbedFn = Callable[..., np.ndarray]
StreamFn = Callable[[str], Iterator[str | Generation]]

HISTORY_TURNS = 6  # 프롬프트에 넣는 이전 대화 메시지 수 (질문·답 3쌍)
NO_LLM_MESSAGE = "GEMINI_API_KEY가 없어 답변을 만들 수 없습니다. 검색 결과만 보여 줍니다."


@dataclass(frozen=True)
class HitsEvent:
    """답변에 근거로 쓸 조문. 인용 번호 [n]은 이 목록의 n번째다."""

    hits: list[SearchHit]


@dataclass(frozen=True)
class DeltaEvent:
    """답변 텍스트 조각."""

    text: str


@dataclass(frozen=True)
class DoneEvent:
    """답변 끝. error가 있으면 answer는 비었거나 중간까지만 있다."""

    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None


AnswerEvent = HitsEvent | DeltaEvent | DoneEvent


@dataclass
class Searcher:
    """인덱스 하나 + 같은 모델의 임베딩 함수 + (선택) LLM 스트림.

    Args:
        index: 열린 VectorIndex.
        embed_fn: 인덱스를 만든 것과 같은 모델의 임베딩 함수.
        profile: 모델 프로필 (쿼리 형식).
        stream: 프롬프트 → 답 조각 스트림. None이면 Gemini(stream_with_usage).
        llm_available: LLM을 쓸 수 있는가. None이면 stream을 줬거나 GEMINI_API_KEY가 있으면 True.
    """

    index: VectorIndex
    embed_fn: EmbedFn
    profile: ModelProfile = E5_STYLE
    stream: StreamFn | None = None
    llm_available: bool | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _law_names: set[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.llm_available is None:
            self.llm_available = self.stream is not None or get_settings().has_gemini_key
        if self.stream is None:
            from ragkit.models.gemini_client import stream_with_usage

            self.stream = stream_with_usage

    @classmethod
    def open(
        cls,
        model: str | None = None,
        checkpoint: Path | None = None,
        backend: str | None = None,
        index_path: Path | None = None,
    ) -> "Searcher":
        """Settings 기본값으로 모델·인덱스를 연다 (`ragkit index`와 같은 파일 규칙).

        Args:
            model: 임베딩 모델 이름 또는 폴더. 기본값 Settings.embedding_model_name.
                MLflow 레지스트리 주소(models:/law-embedder@champion)도 받는다 (extra [mlflow]).
            checkpoint: 파인튜닝한 모델 폴더.
            backend: onnx | torch | st. 기본값 Settings.embedding_backend. 주지 않았는데
                파인튜닝 폴더에 onnx/model.onnx가 없으면 torch로 연다(`ragkit evaluate`와 같은 규칙).
            index_path: 인덱스 파일. 기본값 data/processed/index/<모델 키>.sqlite.

        Raises:
            FileNotFoundError: 인덱스 파일이 없을 때 (만드는 명령을 안내한다).
        """
        from ragkit.embeddings import choose_backend, create_embedding_fn

        model = model or get_settings().embedding_model_name
        key = model_key(model, checkpoint)
        if model.startswith("models:/"):
            model, key = tracking.resolve_model(model)  # 등록 버전의 폴더 + 그 모델로 만든 인덱스 키
        index = VectorIndex.open(index_path or default_index_path(key))
        profile = get_profile(checkpoint or model)
        no_onnx = checkpoint is not None and not (Path(checkpoint) / "onnx" / "model.onnx").exists()
        if backend is None and no_onnx and choose_backend(profile, None) == "onnx" and "torch" in profile.backends:
            backend = "torch"
        embed_fn = create_embedding_fn(model, checkpoint_path=checkpoint, backend=backend)
        return cls(index, embed_fn, profile=profile)

    @property
    def model_key(self) -> str:
        return self.index.model_key

    @property
    def doc_count(self) -> int:
        with self._lock:
            return len(self.index)

    def list_laws(self, theme: str | None = None) -> list[LawInfo]:
        """인덱스에 든 법령 목록. theme을 주면 그 테마만."""
        with self._lock:
            laws = self.index.list_laws()
        return [law for law in laws if theme is None or law.theme == theme]

    def resolve_laws(self, names: list[str]) -> list[str]:
        """법령 이름을 확인한다. 모르는 이름이 있으면 비슷한 이름을 담아 ValueError."""
        if self._law_names is None:
            self._law_names = {law.law_name for law in self.list_laws()}
        for name in names:
            if name not in self._law_names:
                similar = [n for n in sorted(self._law_names) if name in n]
                similar += difflib.get_close_matches(name, self._law_names, n=3)
                hint = f" (비슷한 이름: {', '.join(dict.fromkeys(similar[:5]))})" if similar else ""
                raise ValueError(f"모르는 법령: {name}{hint}")
        return list(names)

    def search(
        self,
        query: str,
        k: int = 5,
        laws: list[str] | None = None,
        *,
        session_id: str | None = None,
        user: str | None = None,
        source: str | None = None,
    ) -> list[SearchHit]:
        """질문과 가까운 조문 상위 k개. laws를 주면 그 법령들 안에서만 찾는다.

        MLflow가 켜져 있으면 검색 한 번이 트레이스 하나로 남는다 (source: 어느 앱에서 왔는지).
        """
        span = tracking.start_span(
            "search", "RETRIEVER", inputs={"query": query, "k": k, "laws": laws},
            session_id=session_id, user=user, tags=_source_tags(source),
        )
        return self._search(query, k, laws, span)

    def _search(self, query: str, k: int, laws: list[str] | None, span: tracking.Span) -> list[SearchHit]:
        """검색하고 span을 닫는다: 질문 임베딩(EMBEDDING) 스팬 + 찾은 조문 본문을 출력으로."""
        try:
            where = {"law_name": self.resolve_laws(laws)} if laws else None
            with self._lock:
                text = self.profile.format_query(query)
                embed_span = tracking.start_span("embed_query", "EMBEDDING", parent=span, inputs={"text": text})
                vector = self.embed_fn([text])[0]
                embed_span.end(outputs={"dim": int(vector.shape[-1])})
                hits = self.index.search(vector, k=k, where=where)
        except Exception as e:
            span.end(error=str(e))
            raise
        span.end(outputs=_documents(hits))
        return hits

    def get(self, doc_id: str) -> SearchHit | None:
        """id로 조문 조각 하나. 없으면 None."""
        with self._lock:
            return self.index.get(doc_id)

    def get_article(self, parent_id: str) -> list[SearchHit]:
        """같은 조의 조각 전체 (코퍼스 순서)."""
        with self._lock:
            return self.index.get_article(parent_id)

    def answer_stream(
        self,
        query: str,
        k: int = 5,
        laws: list[str] | None = None,
        history: list[dict] | None = None,
        *,
        session_id: str | None = None,
        user: str | None = None,
        source: str | None = None,
    ) -> Iterator[AnswerEvent]:
        """검색한 조문만 근거로 LLM 답변을 스트리밍한다.

        MLflow가 켜져 있으면 답변 한 번이 트레이스 하나로 남는다:
        answer(CHAIN) 아래 search(RETRIEVER, 찾은 조문)와 llm(CHAT_MODEL, 프롬프트·답·토큰 수).
        session_id로 같은 대화(노트북)의 트레이스를 묶는다.

        Args:
            query: 질문. 검색에는 이것만 쓴다.
            k: 근거로 줄 조문 수.
            laws: 검색할 법령. None이면 전체.
            history: 이전 대화. 최근 HISTORY_TURNS개만 프롬프트에 넣는다.
            session_id: 대화 세션 (예: 노트북 id).
            user: 사용자.
            source: 어느 앱에서 왔는지 (api · search-cli · mcp).

        Yields:
            HitsEvent 하나 → DeltaEvent 여러 개 → DoneEvent 하나.
        """
        start = time.perf_counter()
        recent = history[-HISTORY_TURNS:] if history else None
        root = tracking.start_span(
            "answer", "CHAIN", inputs={"query": query, "k": k, "laws": laws, "history": recent},
            session_id=session_id, user=user, tags=_source_tags(source),
        )
        llm = tracking.Span()
        done: DoneEvent | None = None
        try:
            search_span = tracking.start_span("search", "RETRIEVER", parent=root,
                                              inputs={"query": query, "k": k, "laws": laws})
            hits = self._search(query, k, laws, search_span)
            yield HitsEvent(hits)
            if not self.llm_available:
                done = DoneEvent(error=NO_LLM_MESSAGE, latency_s=time.perf_counter() - start)
                yield done
                return
            prompt = build_prompt(query, hits, recent)
            llm = tracking.start_span("llm", "CHAT_MODEL", parent=root,
                                      inputs={"messages": [{"role": "user", "content": prompt}]})
            pieces: list[str] = []
            try:
                for item in self.stream(prompt):
                    if isinstance(item, Generation):
                        done = DoneEvent(
                            answer=item.text or "".join(pieces),
                            input_tokens=item.input_tokens,
                            output_tokens=item.output_tokens,
                            latency_s=time.perf_counter() - start,
                        )
                        break
                    pieces.append(item)
                    yield DeltaEvent(item)
            except Exception as e:  # noqa: BLE001 — LLM 오류로 검색 결과까지 잃지 않게 이벤트로 알린다
                done = DoneEvent(answer="".join(pieces), error=f"답변 생성 실패: {e}",
                                 latency_s=time.perf_counter() - start)
            if done is None:
                done = DoneEvent(answer="".join(pieces), latency_s=time.perf_counter() - start)
            llm.end(
                outputs=done.answer,
                attributes={"mlflow.chat.tokenUsage": {
                    "input_tokens": done.input_tokens, "output_tokens": done.output_tokens,
                    "total_tokens": done.input_tokens + done.output_tokens,
                }},
                error=done.error,
            )
            yield done
        finally:
            # 끝까지 갔으면 결과로, 도중에 끊겼으면(클라이언트 연결 종료 등) 오류로 루트를 닫는다
            if done is not None:
                root.end(outputs={"answer": done.answer, "latency_s": done.latency_s}, error=done.error)
            else:
                llm.end(error="중단됨")
                root.end(error="중단됨")

    def answer(
        self,
        query: str,
        k: int = 5,
        laws: list[str] | None = None,
        history: list[dict] | None = None,
        *,
        session_id: str | None = None,
        user: str | None = None,
        source: str | None = None,
    ) -> AnswerResult:
        """answer_stream을 끝까지 모은 결과 (스트리밍이 필요 없는 CLI·MCP·API용)."""
        hits: list[SearchHit] = []
        done = DoneEvent()
        events = self.answer_stream(query, k=k, laws=laws, history=history,
                                    session_id=session_id, user=user, source=source)
        for event in events:
            if isinstance(event, HitsEvent):
                hits = event.hits
            elif isinstance(event, DoneEvent):
                done = event
        return AnswerResult(
            query=query,
            answer=done.answer,
            hits=hits,
            llm_calls=0 if done.error == NO_LLM_MESSAGE else 1,
            input_tokens=done.input_tokens,
            output_tokens=done.output_tokens,
            latency_s=done.latency_s,
            error=done.error,
        )


def _source_tags(source: str | None) -> dict[str, str] | None:
    return {"source": source} if source else None


def _documents(hits: list[SearchHit]) -> list[dict]:
    """RETRIEVER 스팬 출력: MLflow 문서 형식(id·page_content·metadata)으로 찾은 조문 본문 전체."""
    return [
        {
            "id": hit.id,
            "page_content": hit.text,
            "metadata": {"title": hit.metadata.get("title"), "law_name": hit.metadata.get("law_name"),
                         "score": round(float(hit.score), 4)},
        }
        for hit in hits
    ]
