"""Gemini API 클라이언트."""

from google import genai

from src.config import get_settings

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
