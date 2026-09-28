"""Gemini API 클라이언트."""

from dataclasses import dataclass

from google import genai

from ragkit.config import get_settings

_client: genai.Client | None = None


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
    client = _get_client(api_key)

    if model_name is None:
        model_name = get_settings().gemini_model_name

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text


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
    response = client.models.generate_content(
        model=model_name or get_settings().gemini_model_name,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    usage = response.usage_metadata
    return Generation(
        text=response.text or "",
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
    )


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
    response = client.models.count_tokens(
        model=model_name or get_settings().gemini_model_name, contents=text
    )
    return response.total_tokens
