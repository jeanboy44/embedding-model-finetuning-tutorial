"""
Phase 2-2: MCP 서버 이해
=========================

학습 목표:
- MCP(Model Context Protocol)가 무엇인지 이해한다
- LLM이 외부 도구를 사용하는 방식을 이해한다
- 도구(Tool) 함수를 설계하고 구현한다

실행:
    uv run python tutorials/04_product/02_mcp_server.py
"""

import json
from pathlib import Path

from ragkit.config import get_settings
from ragkit.embeddings import create_embedding_fn, format_passages
from ragkit_mcp.legacy import embed_tool, retrieve_tool
from ragkit.retrieval import DocumentStore

# ============================================================
# 1단계: MCP란?
# ============================================================


def step1_what_is_mcp() -> None:
    """MCP 프로토콜의 개념을 설명한다."""
    print("=" * 60)
    print("1단계: MCP(Model Context Protocol)란?")
    print("=" * 60)

    print("""
MCP는 LLM 애플리케이션이 외부 도구/데이터와 상호작용하는 표준 프로토콜이다.

일반적인 LLM 사용:
    사용자 → LLM → 답변 (LLM의 학습 데이터에만 의존)

MCP 사용:
    사용자 → LLM → [MCP 도구 호출] → 외부 데이터/기능 → LLM → 답변

MCP 서버가 제공하는 것:
    1. Tools  - LLM이 호출할 수 있는 함수 (검색, 계산, API 호출 등)
    2. Resources - LLM이 읽을 수 있는 데이터 (파일, DB 등)
    3. Prompts - 미리 정의된 프롬프트 템플릿

우리 프로젝트의 MCP 도구:
    - embed_tool: 텍스트 → 임베딩 벡터
    - retrieve_tool: 쿼리 → 유사 문서 검색
    - rag_tool: 쿼리 → 검색 + 답변 생성
""")


# ============================================================
# 2단계: 도구 함수 설계 원칙
# ============================================================


def step2_tool_design() -> None:
    """좋은 MCP 도구 함수의 설계 원칙을 설명한다."""
    print("\n" + "=" * 60)
    print("2단계: 도구 함수 설계 원칙")
    print("=" * 60)

    print("""
좋은 MCP 도구 함수의 조건:

1. 단일 책임: 하나의 도구는 하나의 작업만 수행
   ✗ def do_everything(query, mode, ...):  # 너무 많은 역할
   ✓ def embed_tool(text):                 # 임베딩만
   ✓ def retrieve_tool(query, k):          # 검색만

2. 명확한 입출력: JSON 직렬화 가능한 타입
   ✗ def embed(text) -> np.ndarray:   # numpy 배열은 JSON 불가
   ✓ def embed(text) -> dict:         # {"shape": ..., "first_5": [...]}

3. 의존성 주입: 무거운 객체는 외부에서 주입
   ✗ def retrieve(query):             # 내부에서 모델 로드
       model = load_model()           # 매번 로드 → 느림
   ✓ def retrieve(query, store, embed_fn):  # 외부에서 주입

4. 에러 처리: LLM이 이해할 수 있는 에러 메시지
   ✓ return {"error": "문서 저장소가 비어있습니다"}
""")


# ============================================================
# 3단계: 도구 함수 실행 데모
# ============================================================


def step3_tool_demo() -> None:
    """MCP 도구 함수를 직접 호출하여 동작을 확인한다."""
    print("\n" + "=" * 60)
    print("3단계: 도구 함수 실행 데모")
    print("=" * 60)

    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)

    # embed_tool 호출
    print("\n[1] embed_tool 호출:")
    result = embed_tool("머신러닝은 데이터에서 패턴을 학습한다", embed_fn)
    print("  입력: '머신러닝은 데이터에서 패턴을 학습한다'")
    print(f"  출력: {json.dumps(result, indent=2, ensure_ascii=False)}")

    # retrieve_tool 호출
    print("\n[2] retrieve_tool 호출:")

    # 먼저 인덱스 구축
    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)

    texts = [d["text"] for d in docs]
    embeddings = embed_fn(format_passages(texts))
    store = DocumentStore()
    store.add_documents(texts, embeddings)

    result = retrieve_tool("임베딩이란 무엇인가?", store, embed_fn, k=3)
    print("  입력: '임베딩이란 무엇인가?'")
    print("  출력:")
    for r in result["results"]:
        print(f"    [{r['score']:.4f}] {r['doc'][:60]}...")


# ============================================================
# 4단계: 도구 함수 직접 만들기
# ============================================================


def step4_build_your_own() -> None:
    """새로운 도구 함수를 만드는 예제를 보여준다."""
    print("\n" + "=" * 60)
    print("4단계: 나만의 도구 함수 만들기")
    print("=" * 60)

    # 예제: 문서 통계 도구
    def stats_tool(store: DocumentStore) -> dict:
        """Tool: 문서 저장소 통계를 반환한다.

        Args:
            store: 문서 저장소.

        Returns:
            문서 수, 임베딩 차원, 평균 문서 길이를 포함하는 딕셔너리.
        """
        avg_len = sum(len(d) for d in store.documents) / len(store.documents)
        return {
            "num_documents": len(store.documents),
            "embedding_dim": store.embeddings.shape[1]
            if store.embeddings is not None
            else 0,
            "avg_document_length": round(avg_len, 1),
        }

    # 예제: 유사 문서 쌍 찾기 도구
    def find_similar_pairs_tool(
        store: DocumentStore,
        embed_fn,
        threshold: float = 0.9,
    ) -> dict:
        """Tool: 유사도가 높은 문서 쌍을 찾는다.

        Args:
            store: 문서 저장소.
            embed_fn: 임베딩 함수.
            threshold: 유사도 임계값.

        Returns:
            유사 문서 쌍 리스트.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(store.embeddings)
        pairs = []
        for i in range(len(store.documents)):
            for j in range(i + 1, len(store.documents)):
                if sims[i][j] >= threshold:
                    pairs.append(
                        {
                            "doc_a": store.documents[i][:50],
                            "doc_b": store.documents[j][:50],
                            "similarity": round(float(sims[i][j]), 4),
                        }
                    )
        return {"threshold": threshold, "pairs": pairs}

    # 실행
    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)

    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)

    texts = [d["text"] for d in docs]
    embeddings = embed_fn(format_passages(texts))
    store = DocumentStore()
    store.add_documents(texts, embeddings)

    print("\n[stats_tool]")
    print(json.dumps(stats_tool(store), indent=2, ensure_ascii=False))

    print("\n[find_similar_pairs_tool]")
    result = find_similar_pairs_tool(store, embed_fn, threshold=0.85)
    print(json.dumps(result, indent=2, ensure_ascii=False))


# ============================================================
# 실행
# ============================================================


def main() -> None:
    """MCP 서버 튜토리얼을 실행한다."""
    step1_what_is_mcp()
    step2_tool_design()
    step3_tool_demo()
    step4_build_your_own()

    print("\n" + "=" * 60)
    print("실습 과제")
    print("=" * 60)
    print("""
1. apps/mcp/src/ragkit_mcp/legacy.py의 코드를 읽고 구조를 파악하세요.
2. 새로운 도구 함수를 하나 만들어보세요:
   - category_search_tool: 특정 카테고리의 문서만 검색
3. 도구 함수의 입출력을 Pydantic 모델로 정의해보세요.
4. (심화) 실제 MCP 서버 apps/mcp/src/ragkit_mcp/server.py(mcp SDK의 MCPServer)를 읽고 ragkit-mcp로 구동해보세요.
""")


if __name__ == "__main__":
    main()
