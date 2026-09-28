"""
Phase 3-2: RAG 에이전트 구축
===============================

학습 목표:
- 검색 + 생성을 결합한 RAG 파이프라인을 구축한다
- 컨텍스트 기반 프롬프트 구성 방법을 익힌다
- RAG의 품질을 좌우하는 요소를 분석한다
- 실패 사례와 해결 방법을 이해한다

사전 준비:
    .env 파일에 GEMINI_API_KEY를 설정하세요.
    (없어도 시뮬레이션 모드로 학습 가능)

실행:
    uv run python tutorials/phase3_agent/02_rag_agent.py
"""

import json
from pathlib import Path

from ragkit.config import get_settings
from ragkit.embeddings import create_embedding_fn, format_passages, format_queries
from ragkit.retrieval import DocumentStore, retrieve

# ============================================================
# 1단계: RAG 파이프라인 단계별 분해
# ============================================================


def step1_rag_decomposition() -> tuple[DocumentStore, list[dict]]:
    """RAG의 각 단계를 분리하여 실행한다.

    Returns:
        (인덱싱된 저장소, 원본 문서 리스트) 튜플.
    """
    print("=" * 60)
    print("1단계: RAG 파이프라인 단계별 분해")
    print("=" * 60)

    print("""
RAG = Retrieval + Augmented + Generation

  사용자 질문
      ↓
  [1. Retrieval] 관련 문서 검색
      ↓
  [2. Augment]   검색 결과를 프롬프트에 삽입
      ↓
  [3. Generate]  LLM이 컨텍스트 기반 답변 생성
      ↓
  답변
""")

    # 준비
    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)

    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)
    embeddings = embed_fn(format_passages([d["text"] for d in docs]))

    store = DocumentStore()
    store.add_documents([d["text"] for d in docs], embeddings)

    query = "RAG 시스템은 어떻게 동작하는가?"

    # Step 1: Retrieval
    print(f"질문: '{query}'\n")
    print("[Step 1: Retrieval]")
    query_emb = embed_fn(format_queries([query]))
    results = retrieve(store, query_emb, k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] {doc[:60]}...")

    # Step 2: Augment
    print("\n[Step 2: Augment — 프롬프트 구성]")
    context = "\n".join([f"- {doc}" for doc, _ in results])
    prompt = (
        f"아래 문서를 참고하여 질문에 한국어로 답변하세요.\n\n"
        f"참고 문서:\n{context}\n\n"
        f"질문: {query}\n\n"
        f"답변:"
    )
    print(f"  프롬프트 길이: {len(prompt)}자")
    print("  프롬프트 미리보기:")
    for line in prompt.split("\n")[:5]:
        print(f"    {line}")
    print("    ...")

    # Step 3: Generate
    print("\n[Step 3: Generate — LLM 응답]")
    if settings.gemini_api_key:
        from ragkit.models import generate_text

        try:
            answer = generate_text(prompt, temperature=0.3, max_output_tokens=300)
            print(f"  {answer[:200]}")
        except (ConnectionError, ValueError, RuntimeError) as e:
            print(f"  API 오류: {e}")
    else:
        print("  [시뮬레이션] API 키 없음 — 아래는 예상 응답:")
        print("  RAG 시스템은 외부 문서를 검색하여 LLM의 답변에 활용하는")
        print("  방식입니다. 먼저 쿼리를 임베딩으로 변환하고, 벡터 검색으로")
        print("  관련 문서를 찾은 후, 이를 프롬프트에 포함시켜 답변합니다.")

    return store, docs


# ============================================================
# 2단계: 프롬프트 변형 실험
# ============================================================


def step2_prompt_variants(store: DocumentStore) -> None:
    """다양한 프롬프트 전략의 효과를 비교한다.

    Args:
        store: 인덱싱된 문서 저장소.
    """
    print("\n" + "=" * 60)
    print("2단계: 프롬프트 변형 실험")
    print("=" * 60)

    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)

    query = "임베딩 모델을 파인튜닝하려면 어떻게 해야 하나요?"
    query_emb = embed_fn(format_queries([query]))
    results = retrieve(store, query_emb, k=3)
    context = "\n".join([f"- {doc}" for doc, _ in results])

    # 다양한 프롬프트 전략
    prompts = {
        "기본": (f"질문: {query}\n답변:"),
        "컨텍스트 포함": (f"참고 문서:\n{context}\n\n질문: {query}\n답변:"),
        "역할 설정": (
            f"당신은 ML 전문가입니다. 초보자도 이해할 수 있게 설명해주세요.\n\n"
            f"참고 문서:\n{context}\n\n질문: {query}\n답변:"
        ),
        "형식 지정": (
            f"참고 문서:\n{context}\n\n질문: {query}\n\n"
            f"아래 형식으로 답변하세요:\n"
            f"1. 핵심 답변 (1문장)\n"
            f"2. 상세 설명 (2-3문장)\n"
            f"3. 참고 사항\n\n답변:"
        ),
        "근거 명시": (
            f"참고 문서:\n{context}\n\n질문: {query}\n\n"
            f"반드시 참고 문서의 내용만을 기반으로 답변하세요.\n"
            f"문서에 없는 내용은 '문서에 해당 정보가 없습니다'라고 답하세요.\n\n답변:"
        ),
    }

    for name, prompt in prompts.items():
        print(f"\n--- {name} ---")
        print(f"  프롬프트 길이: {len(prompt)}자")

        if settings.gemini_api_key:
            from ragkit.models import generate_text

            try:
                answer = generate_text(prompt, temperature=0.3, max_output_tokens=200)
                print(f"  응답: {answer[:150]}...")
            except (ConnectionError, ValueError, RuntimeError) as e:
                print(f"  오류: {e}")
        else:
            print("  [시뮬레이션 모드 — 프롬프트 구조만 확인]")
            print(f"  프롬프트 시작: {prompt[:80]}...")

    print("""
분석:
- '기본': 컨텍스트 없이 LLM 지식에만 의존 → 환각 위험
- '컨텍스트 포함': 검색 결과 기반 → 정확도 향상
- '역할 설정': 답변 스타일 제어
- '형식 지정': 구조화된 출력
- '근거 명시': 환각 방지에 가장 효과적
""")


# ============================================================
# 3단계: RAG vs 순수 LLM 비교
# ============================================================


def step3_rag_vs_llm() -> None:
    """RAG와 순수 LLM의 답변 품질을 비교한다."""
    print("\n" + "=" * 60)
    print("3단계: RAG vs 순수 LLM 비교")
    print("=" * 60)

    print("""
RAG의 장점:
1. 환각 감소 — 검색된 근거 기반 답변
2. 최신 정보 — 문서 업데이트만으로 지식 갱신
3. 출처 추적 — 어떤 문서에서 답변이 나왔는지 확인 가능
4. 도메인 특화 — 사내 문서, 논문 등 특수 지식 활용

RAG의 한계:
1. 검색 품질에 의존 — 관련 문서를 못 찾으면 답변도 부정확
2. 컨텍스트 길이 제한 — 너무 많은 문서를 넣으면 비용/속도 문제
3. 지연시간 증가 — 검색 단계가 추가됨
4. 문서 품질에 의존 — 문서 자체가 틀리면 답변도 틀림

언제 RAG를 쓸 것인가?
- 최신 정보가 필요할 때
- 사내 문서/논문 등 특수 지식이 필요할 때
- 답변의 근거를 제시해야 할 때
- LLM의 학습 데이터에 없는 정보가 필요할 때

언제 순수 LLM이 나은가?
- 일반 상식 수준의 질문
- 창작, 브레인스토밍
- 지연시간이 중요할 때
""")


# ============================================================
# 4단계: 전체 파이프라인을 함수로 조립
# ============================================================


def step4_full_pipeline(store: DocumentStore) -> None:
    """run_rag 함수를 사용하여 전체 파이프라인을 실행한다.

    Args:
        store: 인덱싱된 문서 저장소.
    """
    print("\n" + "=" * 60)
    print("4단계: 전체 파이프라인 (src/agents/rag_agent.py)")
    print("=" * 60)

    settings = get_settings()

    print("""
src/agents/rag_agent.py의 run_rag 함수:

    def run_rag(
        query: str,
        store: DocumentStore,        # 문서 저장소 (DI)
        embed_fn: Callable,          # 임베딩 함수 (DI)
        generate_fn: Callable,       # 생성 함수 (DI)
        k: int = 3,
    ) -> dict:

의존성 주입의 장점:
- embed_fn을 mock으로 바꾸면 모델 없이 테스트 가능
- generate_fn을 바꾸면 다른 LLM(GPT, Claude 등)으로 교체 가능
- store를 바꾸면 다른 벡터 DB 사용 가능
""")

    # 실제 실행 또는 시뮬레이션
    embed_fn = create_embedding_fn(settings.embedding_model_name)

    if settings.gemini_api_key:
        from ragkit.rag import run_rag
        from ragkit.models import generate_text

        queries = [
            "임베딩 파인튜닝은 어떻게 하나요?",
            "RAG에서 검색 품질을 높이는 방법은?",
        ]

        for query in queries:
            print(f"\n질문: {query}")
            try:
                result = run_rag(query, store, embed_fn, generate_text, k=3)
                print(f"검색된 문서: {len(result['retrieved_documents'])}개")
                print(f"답변: {result['answer'][:150]}...")
            except (ConnectionError, ValueError, RuntimeError) as e:
                print(f"오류: {e}")
    else:
        print("\n  [API 키 없음 — run_rag 호출 건너뜀]")
        print("  .env에 GEMINI_API_KEY를 설정하면 실제 동작을 확인할 수 있습니다.")

        # mock으로 테스트하는 예제
        print("\n  mock 함수로 테스트하는 예:")
        print("""
    def mock_generate(prompt: str) -> str:
        return f"[MOCK] 프롬프트 길이 {len(prompt)}자에 대한 응답"

    from ragkit.rag import run_rag
    result = run_rag("질문", store, embed_fn, mock_generate)
""")

        from ragkit.rag import run_rag

        def mock_generate(prompt: str) -> str:
            return f"[MOCK] 프롬프트 길이 {len(prompt)}자에 대한 응답"

        result = run_rag("임베딩이란?", store, embed_fn, mock_generate, k=2)
        print("\n  mock 결과:")
        print(f"    쿼리: {result['query']}")
        print(f"    검색 문서: {len(result['retrieved_documents'])}개")
        print(f"    답변: {result['answer']}")


# ============================================================
# 실행
# ============================================================


def main() -> None:
    """RAG 에이전트 튜토리얼을 실행한다."""
    store, _docs = step1_rag_decomposition()
    step2_prompt_variants(store)
    step3_rag_vs_llm()
    step4_full_pipeline(store)

    print("\n" + "=" * 60)
    print("실습 과제")
    print("=" * 60)
    print("""
1. 프롬프트 템플릿을 직접 설계하고 답변 품질을 비교해보세요.
2. mock_generate를 사용해 run_rag를 테스트하는 코드를 작성하세요.
3. K값(검색 문서 수)을 1, 3, 5, 10으로 바꾸며 답변 품질 변화를 관찰하세요.
4. (심화) 검색 결과에 메타데이터(출처, 카테고리)를 포함하여
   "출처: ..." 형태의 답변을 생성해보세요.
""")


if __name__ == "__main__":
    main()
