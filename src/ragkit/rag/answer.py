"""질문 → (검색) → LLM 답변. 1단계 비교 실습이 같은 결과 형식으로 비교하도록 한 곳에 둔다.

- answer_without_retrieval: 순수 LLM (검색 없음)
- answer_with_rag: 인덱스에서 조문을 찾아 근거로 답변
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ragkit.embeddings.profiles import E5_STYLE
from ragkit.models import Generation, generate_with_usage
from ragkit.retrieval import SearchHit, VectorIndex

GenerateFn = Callable[[str], Generation]
EmbedFn = Callable[..., np.ndarray]

PROMPT_TEMPLATE = """당신은 한국 법령 안내 도우미입니다.
아래 [조문]만 근거로 질문에 답하세요. 조문에 근거가 없으면 "제공된 조문으로는 알 수 없습니다"라고 답하세요.
답의 끝에 근거로 쓴 조문 번호를 [1]처럼 적으세요.

[조문]
{context}

[질문]
{query}"""


@dataclass(frozen=True)
class AnswerResult:
    """답변 한 건과 비용 측정값."""

    query: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


def build_prompt(query: str, hits: list[SearchHit]) -> str:
    """검색한 조문을 번호를 붙여 프롬프트에 넣는다."""
    context = "\n\n".join(
        f"[{i}] {hit.metadata.get('title', hit.id)}\n{hit.text}" for i, hit in enumerate(hits, 1)
    )
    return PROMPT_TEMPLATE.format(context=context, query=query)


def answer_without_retrieval(query: str, *, generate: GenerateFn = generate_with_usage) -> AnswerResult:
    """검색 없이 질문만 LLM에 보낸다 (순수 LLM 기준선)."""
    start = time.perf_counter()
    gen = generate(query)
    return AnswerResult(
        query=query,
        answer=gen.text,
        llm_calls=1,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
        latency_s=time.perf_counter() - start,
    )


def answer_with_rag(
    query: str,
    index: VectorIndex,
    embed_fn: EmbedFn,
    *,
    generate: GenerateFn = generate_with_usage,
    k: int = 5,
    where: dict | None = None,
    format_query: Callable[[str], str] | None = None,
) -> AnswerResult:
    """인덱스에서 상위 k개 조문을 찾아 그것만 근거로 답한다.

    Args:
        query: 사용자 질문.
        index: build_index / VectorIndex.open으로 연 인덱스.
        embed_fn: 인덱스를 만든 것과 같은 모델의 임베딩 함수.
        generate: 프롬프트 → Generation. 기본값 Gemini.
        k: 프롬프트에 넣을 조문 수.
        where: 메타데이터 필터 (VectorIndex.search와 같음).
        format_query: 모델 프로필의 쿼리 형식. None이면 e5 형식("query: ").

    Returns:
        답변, 검색 결과, LLM 호출 수·토큰 수·지연 시간 (검색 시간 포함).
    """
    start = time.perf_counter()
    format_query = format_query or E5_STYLE.format_query
    hits = index.search(embed_fn([format_query(query)])[0], k=k, where=where)
    gen = generate(build_prompt(query, hits))
    return AnswerResult(
        query=query,
        answer=gen.text,
        hits=hits,
        llm_calls=1,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
        latency_s=time.perf_counter() - start,
    )
