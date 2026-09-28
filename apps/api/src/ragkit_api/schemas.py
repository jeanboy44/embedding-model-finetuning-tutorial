"""요청·응답 스키마. apps/web의 src/api/types.ts와 1:1로 맞춘다."""

from pydantic import BaseModel, Field

from ragkit.retrieval import LawInfo, SearchHit


class Hit(BaseModel):
    """조문 조각 하나 (검색 결과, 인용)."""

    id: str
    parent_id: str | None = None
    title: str | None = None
    text: str
    score: float = 0.0
    law_name: str | None = None
    law_type: str | None = None
    theme: str | None = None
    article_no: str | None = None
    article_title: str | None = None
    chapter: str | None = None
    source_url: str | None = None

    @classmethod
    def from_hit(cls, hit: SearchHit) -> "Hit":
        return cls.model_validate(hit.to_dict())


class Law(BaseModel):
    law_name: str
    law_type: str
    theme: str
    doc_count: int

    @classmethod
    def from_info(cls, info: LawInfo) -> "Law":
        return cls(law_name=info.law_name, law_type=info.law_type, theme=info.theme, doc_count=info.doc_count)


class Health(BaseModel):
    status: str = "ok"
    model_key: str
    doc_count: int
    llm_available: bool


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=50)
    laws: list[str] | None = None


class SearchResponse(BaseModel):
    hits: list[Hit]


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AnswerRequest(SearchRequest):
    history: list[Turn] | None = None


class AnswerResponse(BaseModel):
    answer: str
    hits: list[Hit]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None


class NotebookCreate(BaseModel):
    title: str = Field("제목 없는 노트북", min_length=1)
    laws: list[str] = []


class NotebookUpdate(BaseModel):
    title: str | None = Field(None, min_length=1)
    laws: list[str] | None = None


class Notebook(BaseModel):
    id: str
    title: str
    laws: list[str]
    created_at: str
    updated_at: str


class Message(BaseModel):
    id: str
    role: str
    content: str
    citations: list[Hit] = []
    error: str | None = None
    created_at: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=20)


class NoteCreate(BaseModel):
    title: str = ""
    content: str = ""
    citations: list[Hit] = []


class NoteUpdate(BaseModel):
    title: str | None = None
    content: str | None = None


class Note(BaseModel):
    id: str
    title: str
    content: str
    citations: list[Hit] = []
    created_at: str
    updated_at: str
