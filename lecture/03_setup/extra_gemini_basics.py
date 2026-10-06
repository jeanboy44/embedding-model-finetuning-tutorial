"""
선택 실습 (3교시): Gemini API 기초
=====================

학습 목표:
- Gemini API를 호출하여 텍스트를 생성한다
- 프롬프트 엔지니어링의 기초를 익힌다
- 검색 결과를 컨텍스트로 사용하는 패턴을 이해한다

사전 준비:
    .env 파일에 GEMINI_API_KEY를 설정하세요 (없으면 호출 단계는 코드만 보여 주고 건너뛴다).
    무료 등급은 하루 호출 수가 적다(flash-lite 20회). 이 스크립트는 한 번 실행에 4회 호출한다
    (기본 질문 · 컨텍스트 기반 · temperature 2개).

실행:
    uv run python lecture/03_setup/extra_gemini_basics.py
"""

from google.genai import errors

from ragkit.config import get_settings

# ============================================================
# 1. API 키 확인
# ============================================================


def step1_check_api_key() -> bool:
    """Gemini API 키가 설정되어 있는지 확인한다.

    Returns:
        API 키가 설정되어 있으면 True.
    """
    print("=" * 60)
    print("1. API 키 확인")
    print("=" * 60)

    if not get_settings().has_gemini_key:
        print("""
  API 키가 설정되지 않았습니다.

  설정 방법:
  1. .env 파일을 열고 (없으면 .env.example을 복사)
  2. GEMINI_API_KEY=your_key_here 를 입력
  3. API 키는 https://aistudio.google.com/apikey 에서 발급

  이 튜토리얼은 API 키 없이도 구조를 학습할 수 있습니다.
  (실제 API 호출 단계는 건너뜁니다)
""")
        return False

    print("  API 키가 설정되어 있습니다.")
    return True


# ============================================================
# 2. 프롬프트 엔지니어링 기초
# ============================================================


def step2_prompt_engineering() -> None:
    """프롬프트 설계 패턴을 설명한다."""
    print("\n" + "=" * 60)
    print("2. 프롬프트 엔지니어링 기초")
    print("=" * 60)

    print("""
프롬프트는 LLM에게 주는 "지시문"이다. 좋은 프롬프트의 요소:

1. 역할 설정 (System Prompt)
   "당신은 ML 전문가입니다. 초보자에게 쉽게 설명해주세요."

2. 컨텍스트 제공
   "아래 문서를 참고하여 답변하세요: ..."

3. 구체적 지시
   "3문장 이내로 답변하세요." / "JSON 형식으로 출력하세요."

4. 예시 제공 (Few-shot)
   "예시: 질문: X, 답변: Y"
""")

    # 프롬프트 템플릿 예시
    templates = {
        "기본 QA": ("질문: {query}\n답변:"),
        "컨텍스트 기반 QA": (
            "아래 문서를 참고하여 질문에 답변하세요.\n\n"
            "문서:\n{context}\n\n"
            "질문: {query}\n\n"
            "답변:"
        ),
        "쿼리 확장": (
            "다음 검색 쿼리를 3가지 다른 표현으로 변환하세요.\n"
            "원본: {query}\n"
            "변환 (한 줄에 하나씩):"
        ),
        "요약": ("아래 텍스트를 2문장으로 요약하세요.\n\n{text}\n\n요약:"),
    }

    print("\n프롬프트 템플릿 예시:")
    for name, template in templates.items():
        print(f"\n--- {name} ---")
        # 예시 값으로 채우기
        filled = template.format(
            query="임베딩이란 무엇인가?",
            context="- 임베딩은 텍스트를 벡터로 변환하는 기법이다.\n- 코사인 유사도로 유사성을 측정한다.",
            text="트랜스포머는 셀프 어텐션 기반 아키텍처로 병렬 처리가 가능하다.",
        )
        print(filled)


# ============================================================
# 3. Gemini API 호출
# ============================================================


def step3_api_call(has_key: bool) -> None:
    """Gemini API를 실제로 호출하거나 시뮬레이션한다.

    Args:
        has_key: API 키가 있으면 True.
    """
    print("\n" + "=" * 60)
    print("3. Gemini API 호출")
    print("=" * 60)

    if not has_key:
        print("\n  [시뮬레이션 모드 — API 키 없음]")
        print("\n  실제 호출 코드:")
        print("""
    from ragkit.models import generate_text

    # 기본 호출
    answer = generate_text("임베딩이란 무엇인가?")

    # 파라미터 조정
    answer = generate_text(
        "임베딩이란 무엇인가?",
        temperature=0.3,        # 낮을수록 결정적
        max_output_tokens=512,  # 출력 길이 제한
    )
""")
        print("  시뮬레이션 응답:")
        print("  '임베딩은 텍스트나 이미지 등의 데이터를 고정 길이의")
        print("   밀집 벡터로 변환하는 기법입니다...'")
        return

    from ragkit.models import generate_text

    prompts = [
        ("기본 질문", "임베딩이란 무엇인가? 2문장으로 설명해주세요."),
        (
            "컨텍스트 기반",
            (
                "아래 문서를 참고하여 답변하세요.\n\n"
                "문서: RAG는 외부 지식 베이스에서 관련 문서를 검색한 후, "
                "이를 컨텍스트로 사용하여 LLM이 답변을 생성하는 방식이다.\n\n"
                "질문: RAG의 장점은 무엇인가?\n답변:"
            ),
        ),
    ]

    for name, prompt in prompts:
        print(f"\n[{name}]")
        print(f"  프롬프트: {prompt[:60]}...")
        try:
            answer = generate_text(prompt, temperature=0.3, max_output_tokens=200)
            print(f"  응답: {answer[:200]}")
        except (ConnectionError, ValueError, RuntimeError, errors.APIError) as e:
            print(f"  오류: {e}")


# ============================================================
# 4. Temperature 실험
# ============================================================


def step4_temperature(has_key: bool) -> None:
    """Temperature 파라미터의 효과를 실험한다.

    Args:
        has_key: API 키가 있으면 True.
    """
    print("\n" + "=" * 60)
    print("4. Temperature 실험")
    print("=" * 60)

    print("""
Temperature는 생성의 "창의성"을 조절한다:
  - 0.0: 가장 확률 높은 토큰만 선택 (결정적, 반복적)
  - 0.3: 약간의 다양성 (사실 기반 QA에 적합)
  - 0.7: 균형잡힌 다양성 (일반 대화에 적합)
  - 1.0: 높은 다양성 (창작, 브레인스토밍에 적합)

RAG 시스템에서는 보통 0.1~0.3을 사용한다.
(검색된 문서에 기반한 정확한 답변이 중요하므로)
""")

    if not has_key:
        print("  [API 키가 없어 실제 실험은 건너뜁니다]")
        return

    from ragkit.models import generate_text

    prompt = "머신러닝을 한 문장으로 설명해주세요."
    temperatures = [0.0, 1.0]  # 무료 등급 호출 수를 아끼려 양 끝 두 값만 비교한다

    for temp in temperatures:
        print(f"\n  Temperature={temp}:")
        try:
            answer = generate_text(prompt, temperature=temp, max_output_tokens=100)
            print(f"    {answer[:100]}")
        except (ConnectionError, ValueError, RuntimeError, errors.APIError) as e:
            print(f"    오류: {e}")


# ============================================================
# 실행
# ============================================================


def main() -> None:
    """Gemini API 연동 튜토리얼을 실행한다."""
    has_key = step1_check_api_key()
    step2_prompt_engineering()
    step3_api_call(has_key)
    step4_temperature(has_key)

    print("\n" + "=" * 60)
    print("실습 과제")
    print("=" * 60)
    print("""
1. .env에 GEMINI_API_KEY를 설정하고 실제 API를 호출해보세요.
2. 자신만의 프롬프트 템플릿을 만들어보세요.
3. temperature를 바꿔가며 같은 질문에 대한 답변 변화를 관찰하세요 (호출 수 한도에 주의).
4. src/ragkit/models/gemini_client.py의 generate_text 함수를 읽고
   새로운 파라미터(top_p, top_k)를 추가해보세요.
""")


if __name__ == "__main__":
    main()
