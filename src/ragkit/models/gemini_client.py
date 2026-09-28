"""Gemini API 클라이언트."""

import itertools
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

from google import genai
from google.genai import errors

from ragkit.config import get_settings

_client: genai.Client | None = None

# 무료 등급에서 흔한 일시 오류: 429(한도 초과), 500/503(서버 과부하)
RETRY_CODES = {429, 500, 503}
MAX_RETRIES = 4
_sleep = time.sleep

T = TypeVar("T")


def _with_retry(call: Callable[[], T]) -> T:
    """일시 오류면 2, 4, 8, 16초 기다리며 다시 시도한다. 그 밖의 오류는 바로 올린다."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return call()
        except errors.APIError as e:
            if e.code not in RETRY_CODES or attempt == MAX_RETRIES:
                raise
            _sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def _get_client(api_key: str | None = None) -> genai.Client:
    """Gemini API 클라이언트를 반환한다 (최초 1회 생성).

    Args:
        api_key: Gemini API 키. None이면 Settings에서 로드.

    Returns:
        genai.Client 인스턴스.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = genai.Client(api_key=api_key or settings.gemini_api_key)
    return _client


def generate_text(
    prompt: str,
    *,
    model_name: str | None = None,
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
    api_key: str | None = None,
) -> str:
    """Gemini로 텍스트를 생성한다.

    Args:
        prompt: 입력 프롬프트.
        model_name: Gemini 모델 이름. None이면 Settings 기본값 사용.
        temperature: 생성 온도 (0.0~1.0).
        max_output_tokens: 최대 출력 토큰 수.
        api_key: Gemini API 키. None이면 Settings에서 로드.

    Returns:
        생성된 텍스트.
    """
    return generate_with_usage(
        prompt,
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
    ).text


@dataclass(frozen=True)
class Generation:
    """생성 결과와 토큰 사용량 (비용 비교용)."""

    text: str
    input_tokens: int
    output_tokens: int


def generate_with_usage(
    prompt: str,
    *,
    model_name: str | None = None,
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
    api_key: str | None = None,
) -> Generation:
    """Gemini로 텍스트를 생성하고 입력·출력 토큰 수를 함께 반환한다.

    Args:
        prompt: 입력 프롬프트.
        model_name: Gemini 모델 이름. None이면 Settings 기본값 사용.
        temperature: 생성 온도 (0.0~1.0).
        max_output_tokens: 최대 출력 토큰 수.
        api_key: Gemini API 키. None이면 Settings에서 로드.

    Returns:
        답변 텍스트와 토큰 사용량.
    """
    client = _get_client(api_key)
    response = _with_retry(
        lambda: client.models.generate_content(
            model=model_name or get_settings().gemini_model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                # 도구 호출을 쓰지 않으므로 끈다 (SDK 경고 메시지도 사라진다)
                automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
    )
    usage = response.usage_metadata
    return Generation(
        text=response.text or "",
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
    )


def stream_with_usage(
    prompt: str,
    *,
    model_name: str | None = None,
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
    api_key: str | None = None,
) -> Iterator[str | Generation]:
    """Gemini 답변을 조각으로 받아 내보낸다 (화면에 글자가 흘러나오게).

    요청은 첫 조각을 받을 때 실제로 나가므로, 일시 오류 재시도는 첫 조각 전까지만 한다.
    조각을 내보낸 뒤 다시 시도하면 같은 글이 두 번 나오기 때문이다.

    Args:
        prompt: 입력 프롬프트.
        model_name: Gemini 모델 이름. None이면 Settings 기본값 사용.
        temperature: 생성 온도 (0.0~1.0).
        max_output_tokens: 최대 출력 토큰 수.
        api_key: Gemini API 키. None이면 Settings에서 로드.

    Yields:
        텍스트 조각(str)들, 마지막에 전체 텍스트와 토큰 수를 담은 Generation 하나.
    """
    client = _get_client(api_key)

    def open_stream() -> tuple[Iterator, object]:
        stream = iter(
            client.models.generate_content_stream(
                model=model_name or get_settings().gemini_model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        )
        return stream, next(stream, None)

    stream, first = _with_retry(open_stream)
    pieces: list[str] = []
    input_tokens = output_tokens = 0
    for chunk in itertools.chain([first] if first is not None else [], stream):
        usage = chunk.usage_metadata
        if usage is not None:
            input_tokens = usage.prompt_token_count or input_tokens
            output_tokens = usage.candidates_token_count or output_tokens
        if chunk.text:
            pieces.append(chunk.text)
            yield chunk.text
    yield Generation(text="".join(pieces), input_tokens=input_tokens, output_tokens=output_tokens)


def count_tokens(
    text: str, *, model_name: str | None = None, api_key: str | None = None
) -> int:
    """LLM 토크나이저 기준 토큰 수를 센다 (API 호출 1회, 생성 비용 없음).

    Args:
        text: 토큰 수를 셀 텍스트.
        model_name: Gemini 모델 이름. None이면 Settings 기본값 사용.
        api_key: Gemini API 키. None이면 Settings에서 로드.

    Returns:
        토큰 수.
    """
    client = _get_client(api_key)
    response = _with_retry(
        lambda: client.models.count_tokens(
            model=model_name or get_settings().gemini_model_name, contents=text
        )
    )
    return response.total_tokens
