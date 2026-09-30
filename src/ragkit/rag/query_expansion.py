"""LLM 쿼리 확장: 일상 말투 질문을 법령 용어 검색어로 바꿔 원문 뒤에 붙인다.

파인튜닝 없이 검색 누락을 메우는 방법이다. 대신 질문마다 LLM을 한 번 부르므로
비용·지연이 질문 수에 비례한다(파인튜닝은 이 비용을 학습 1회로 옮긴다).

같은 질문을 다시 부르지 않도록 확장 결과를 JSONL 캐시에 한 줄씩 쌓는다.
무료 등급 한도로 중간에 멈춰도 이어서 받을 수 있고, 강의에서는 캐시 파일을 배포해
API 키 없이도 같은 평가를 돌린다.
"""

import json
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ragkit.config import get_settings
from ragkit.models import Generation, generate_with_usage

GenerateFn = Callable[[str], Generation]

PROMPT_TEMPLATE = """당신은 한국 법령 조문 검색을 돕습니다.
아래 질문은 일상 말투입니다. 관련 조문을 찾기 쉽도록 질문의 뜻을 법령에 쓰이는 말로 바꾼 검색어를 한 줄로 쓰세요.
- 관련 법령 이름과 법률 용어(예: 해고, 퇴직급여, 유급휴일)를 넣으세요
- 설명이나 답변은 쓰지 말고 검색어만 쓰세요

[질문]
{query}"""


class ExpansionUnavailable(RuntimeError):
    """캐시에 없는 질문이 있는데 LLM을 부를 수 없을 때 (API 키 없음)."""


@dataclass(frozen=True)
class Expansion:
    """질문 하나의 확장 결과와 그때 든 비용."""

    query: str
    expansion: str
    llm: str
    input_tokens: int
    output_tokens: int
    latency_s: float


def expansion_prompt(query: str) -> str:
    """질문을 법령 용어 검색어로 바꾸라는 프롬프트."""
    return PROMPT_TEMPLATE.format(query=query)


def search_text(query: str, expansion: str) -> str:
    """검색에 쓸 문장. 확장어만 쓰면 원래 질문의 뜻이 빠질 수 있어 원문 뒤에 붙인다."""
    return f"{query} {expansion}".strip()


def default_cache_path(llm: str) -> Path:
    """확장 캐시 기본 위치: data/processed/query_expansion/{llm}.jsonl."""
    return get_settings().data_dir / "processed" / "query_expansion" / f"{llm}.jsonl"


def load_cache(path: Path, *, llm: str | None = None) -> dict[str, Expansion]:
    """캐시 파일을 질문 → Expansion으로 읽는다. llm을 주면 그 모델의 결과만."""
    if not path.exists():
        return {}
    cache: dict[str, Expansion] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = Expansion(**json.loads(line))
            if llm is None or item.llm == llm:
                cache[item.query] = item
    return cache


def expand_queries(
    queries: Iterable[str],
    cache_path: Path,
    *,
    generate: GenerateFn | None = None,
    llm: str | None = None,
    min_interval_s: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Expansion]:
    """질문들을 확장한다. 캐시에 있는 질문은 LLM을 부르지 않는다.

    Args:
        queries: 질문 목록 (중복은 한 번만 확장).
        cache_path: JSONL 캐시 파일. 새 결과를 받을 때마다 한 줄씩 덧붙인다.
        generate: 프롬프트 → Generation. None이면 Gemini(temperature 0)를 쓴다.
        llm: 캐시에 기록할 LLM 이름. None이면 설정의 gemini_model_name.
        min_interval_s: LLM 호출 사이 최소 간격(초). 무료 등급 분당 한도용 (rpm 15 → 4초).
        sleep: 기다리는 함수 (테스트용).
        clock: 시각 함수 (테스트용).
        on_progress: (새로 받은 수, 받아야 할 수)를 받는 콜백.

    Returns:
        질문 → Expansion.

    Raises:
        ExpansionUnavailable: 캐시에 없는 질문이 있는데 generate도 API 키도 없을 때.
    """
    settings = get_settings()
    llm = llm or settings.gemini_model_name
    wanted = list(dict.fromkeys(queries))
    cache = load_cache(cache_path, llm=llm)
    missing = [q for q in wanted if q not in cache]
    if missing and generate is None:
        if not settings.gemini_api_key:
            raise ExpansionUnavailable(
                f"캐시({cache_path})에 없는 질문이 {len(missing)}개인데 GEMINI_API_KEY가 없습니다. "
                "강의용 캐시 파일을 받거나 .env에 키를 넣으세요."
            )
        generate = _gemini(llm)

    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_call: float | None = None
    with cache_path.open("a", encoding="utf-8") if missing else nullcontext() as out:
        for done, query in enumerate(missing, start=1):
            if last_call is not None:
                wait = min_interval_s - (clock() - last_call)
                if wait > 0:
                    sleep(wait)
            last_call = start = clock()
            generation = generate(expansion_prompt(query))
            item = Expansion(
                query=query,
                expansion=" ".join(generation.text.split()),
                llm=llm,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                latency_s=clock() - start,
            )
            out.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            out.flush()
            cache[query] = item
            if on_progress:
                on_progress(done, len(missing))
    return {q: cache[q] for q in wanted}


def expansion_summary(expansions: list[Expansion]) -> dict:
    """비교표에 넣을 질문당 비용: LLM 호출 수, 평균 토큰, 평균 지연."""
    return {
        "llm": expansions[0].llm if expansions else None,
        "llm_calls_per_query": 1,
        "input_tokens_mean": float(np.mean([e.input_tokens for e in expansions])),
        "output_tokens_mean": float(np.mean([e.output_tokens for e in expansions])),
        "latency_s_mean": float(np.mean([e.latency_s for e in expansions])),
    }


def _gemini(llm: str) -> GenerateFn:
    # 같은 질문이면 같은 검색어가 나오도록 temperature 0
    return lambda prompt: generate_with_usage(
        prompt, model_name=llm, temperature=0.0, max_output_tokens=128
    )
