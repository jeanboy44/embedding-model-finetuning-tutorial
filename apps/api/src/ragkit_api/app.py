"""FastAPI 앱: 검색·답변(SSE)·노트북. web(apps/web)의 유일한 백엔드다.

    app = create_app()                           # 시작할 때 Searcher.open()으로 모델·인덱스 로드
    app = create_app(searcher=fake, store=store) # 테스트: 가짜 주입

엔드포인트는 모두 /api 아래에 둔다. web_dist를 주면 빌드한 web을 / 에서 함께 제공한다.
"""

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from ragkit.config import get_settings
from ragkit.service import AnswerEvent, DeltaEvent, DoneEvent, HitsEvent, Searcher
from ragkit_api.schemas import (
    AnswerRequest,
    AnswerResponse,
    ChatRequest,
    Health,
    Hit,
    Law,
    Message,
    Note,
    Notebook,
    NotebookCreate,
    NotebookUpdate,
    NoteCreate,
    NoteUpdate,
    SearchRequest,
    SearchResponse,
)
from ragkit_api.store import NotebookStore


def sse(event: str, data: dict) -> str:
    """Server-Sent Events 한 건."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _event_payload(event: AnswerEvent) -> tuple[str, dict]:
    if isinstance(event, HitsEvent):
        return "hits", {"hits": [Hit.from_hit(h).model_dump() for h in event.hits]}
    if isinstance(event, DeltaEvent):
        return "delta", {"text": event.text}
    return "done", {"answer": event.answer, "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens, "latency_s": event.latency_s, "error": event.error}


def default_db_path() -> Path:
    return get_settings().data_dir / "app" / "notebooks.sqlite"


def create_app(
    searcher: Searcher | None = None,
    store: NotebookStore | None = None,
    *,
    open_kwargs: dict | None = None,
    web_dist: Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """API 앱을 만든다.

    Args:
        searcher: 검색기. None이면 앱이 시작할 때 Searcher.open(**open_kwargs)로 연다.
        store: 노트북 저장소. None이면 data/app/notebooks.sqlite.
        open_kwargs: Searcher.open 인자 (model, checkpoint, backend, index_path).
        web_dist: 빌드한 web 폴더(apps/web/dist). 주면 / 에서 함께 제공한다.
        cors_origins: 다른 주소에서 띄운 web을 허용할 출처. 기본 Vite 개발 서버.
    """
    state: dict = {"searcher": searcher, "store": store}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["searcher"] is None:
            state["searcher"] = Searcher.open(**(open_kwargs or {}))
        if state["store"] is None:
            state["store"] = NotebookStore(default_db_path())
        yield

    app = FastAPI(title="ragkit API", description="법령 검색 · 근거 답변 · 노트북", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_searcher() -> Searcher:
        return state["searcher"]

    def get_store() -> NotebookStore:
        return state["store"]

    def check_laws(laws: list[str] | None) -> list[str] | None:
        try:
            return get_searcher().resolve_laws(laws) if laws else None
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    def notebook_or_404(notebook_id: str) -> dict:
        notebook = get_store().get_notebook(notebook_id)
        if notebook is None:
            raise HTTPException(404, f"노트북이 없습니다: {notebook_id}")
        return notebook

    # 검색
    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        searcher = get_searcher()
        return Health(model_key=searcher.model_key, doc_count=searcher.doc_count,
                      llm_available=bool(searcher.llm_available))

    @app.get("/api/laws", response_model=list[Law])
    def laws(theme: str | None = None) -> list[Law]:
        return [Law.from_info(info) for info in get_searcher().list_laws(theme=theme)]

    @app.get("/api/docs/{doc_id}", response_model=Hit)
    def get_doc(doc_id: str) -> Hit:
        hit = get_searcher().get(doc_id)
        if hit is None:
            raise HTTPException(404, f"조문이 없습니다: {doc_id}")
        return Hit.from_hit(hit)

    @app.get("/api/articles/{parent_id}", response_model=list[Hit])
    def get_article(parent_id: str) -> list[Hit]:
        hits = get_searcher().get_article(parent_id)
        if not hits:
            raise HTTPException(404, f"조문이 없습니다: {parent_id}")
        return [Hit.from_hit(h) for h in hits]

    @app.post("/api/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        hits = get_searcher().search(req.query, k=req.k, laws=check_laws(req.laws))
        return SearchResponse(hits=[Hit.from_hit(h) for h in hits])

    @app.post("/api/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        history = [t.model_dump() for t in req.history] if req.history else None
        result = get_searcher().answer(req.query, k=req.k, laws=check_laws(req.laws), history=history)
        return AnswerResponse(
            answer=result.answer, hits=[Hit.from_hit(h) for h in result.hits],
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            latency_s=result.latency_s, error=result.error,
        )

    @app.post("/api/answer/stream")
    def answer_stream(req: AnswerRequest) -> StreamingResponse:
        laws = check_laws(req.laws)
        history = [t.model_dump() for t in req.history] if req.history else None

        def events() -> Iterator[str]:
            for event in get_searcher().answer_stream(req.query, k=req.k, laws=laws, history=history):
                yield sse(*_event_payload(event))

        return StreamingResponse(events(), media_type="text/event-stream")

    # 노트북
    @app.get("/api/notebooks", response_model=list[Notebook])
    def list_notebooks() -> list[dict]:
        return get_store().list_notebooks()

    @app.post("/api/notebooks", response_model=Notebook, status_code=201)
    def create_notebook(req: NotebookCreate) -> dict:
        return get_store().create_notebook(req.title, check_laws(req.laws) or [])

    @app.get("/api/notebooks/{notebook_id}", response_model=Notebook)
    def get_notebook(notebook_id: str) -> dict:
        return notebook_or_404(notebook_id)

    @app.patch("/api/notebooks/{notebook_id}", response_model=Notebook)
    def update_notebook(notebook_id: str, req: NotebookUpdate) -> dict:
        notebook_or_404(notebook_id)
        laws = None if req.laws is None else (check_laws(req.laws) or [])
        return get_store().update_notebook(notebook_id, title=req.title, laws=laws)

    @app.delete("/api/notebooks/{notebook_id}", status_code=204)
    def delete_notebook(notebook_id: str) -> None:
        if not get_store().delete_notebook(notebook_id):
            raise HTTPException(404, f"노트북이 없습니다: {notebook_id}")

    @app.get("/api/notebooks/{notebook_id}/messages", response_model=list[Message])
    def list_messages(notebook_id: str) -> list[dict]:
        notebook_or_404(notebook_id)
        return get_store().list_messages(notebook_id)

    @app.delete("/api/notebooks/{notebook_id}/messages", status_code=204)
    def clear_messages(notebook_id: str) -> None:
        notebook_or_404(notebook_id)
        get_store().clear_messages(notebook_id)

    @app.post("/api/notebooks/{notebook_id}/chat")
    def chat(notebook_id: str, req: ChatRequest) -> StreamingResponse:
        """노트북의 법령 안에서 답을 스트리밍하고, 질문과 답을 대화 기록에 저장한다."""
        notebook = notebook_or_404(notebook_id)
        laws = check_laws(notebook["laws"])
        store = get_store()
        history = [{"role": m["role"], "content": m["content"]}
                   for m in store.list_messages(notebook_id) if not m["error"]]
        store.add_message(notebook_id, "user", req.query)

        def events() -> Iterator[str]:
            hits: list[dict] = []
            for event in get_searcher().answer_stream(req.query, k=req.k, laws=laws, history=history):
                name, data = _event_payload(event)
                if isinstance(event, HitsEvent):
                    hits = data["hits"]
                if isinstance(event, DoneEvent):
                    saved = store.add_message(notebook_id, "assistant", event.answer, citations=hits,
                                              error=event.error)
                    data["message_id"] = saved["id"]
                yield sse(name, data)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/notebooks/{notebook_id}/notes", response_model=list[Note])
    def list_notes(notebook_id: str) -> list[dict]:
        notebook_or_404(notebook_id)
        return get_store().list_notes(notebook_id)

    @app.post("/api/notebooks/{notebook_id}/notes", response_model=Note, status_code=201)
    def create_note(notebook_id: str, req: NoteCreate) -> dict:
        notebook_or_404(notebook_id)
        citations = [c.model_dump() for c in req.citations]
        return get_store().create_note(notebook_id, req.title, req.content, citations)

    @app.patch("/api/notebooks/{notebook_id}/notes/{note_id}", response_model=Note)
    def update_note(notebook_id: str, note_id: str, req: NoteUpdate) -> dict:
        note = get_store().update_note(notebook_id, note_id, req.title, req.content)
        if note is None:
            raise HTTPException(404, f"노트가 없습니다: {note_id}")
        return note

    @app.delete("/api/notebooks/{notebook_id}/notes/{note_id}", status_code=204)
    def delete_note(notebook_id: str, note_id: str) -> None:
        if not get_store().delete_note(notebook_id, note_id):
            raise HTTPException(404, f"노트가 없습니다: {note_id}")

    if web_dist is not None:
        _mount_web(app, Path(web_dist))
    return app


def _mount_web(app: FastAPI, dist: Path) -> None:
    """빌드한 SPA를 제공한다. 파일이 없는 경로(/notebooks/..)는 index.html로 돌려 화면 라우터가 처리한다."""
    index_html = dist / "index.html"
    if not index_html.exists():
        raise FileNotFoundError(f"web 빌드가 없습니다: {index_html}\n먼저 빌드하세요: cd apps/web && pnpm build")
    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def web(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404)
        file = (dist / path).resolve()
        if path and file.is_file() and file.is_relative_to(root):
            return FileResponse(file)
        return FileResponse(index_html)
