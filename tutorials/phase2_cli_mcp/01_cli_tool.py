"""
Phase 2-1: CLI 도구 개발
========================

학습 목표:
- cyclopts로 타입 힌트 기반 CLI를 구축한다
- ML 모델을 CLI로 감싸는 패턴을 익힌다
- 의존성 주입으로 테스트 가능한 CLI를 만든다

실행:
    uv run python tutorials/phase2_cli_mcp/01_cli_tool.py
"""

import json
from pathlib import Path

from src.config import get_settings
from src.embeddings import create_embedding_fn, format_passages, format_queries
from src.retrieval import DocumentStore, retrieve

# ============================================================
# 1단계: CLI가 왜 필요한가?
# ============================================================
#
# ML 모델을 개발하면 다양한 사용 시나리오가 생긴다:
#   - 터미널에서 빠르게 쿼리 테스트
#   - 배치 처리 스크립트에서 호출
#   - CI/CD 파이프라인에서 자동 평가
#   - 다른 개발자가 모델을 사용
#
# CLI는 이 모든 시나리오를 하나의 인터페이스로 해결한다.


# ============================================================
# 2단계: 함수부터 만들기
# ============================================================
# CLI의 핵심은 "함수를 먼저 만들고, CLI는 진입점만 제공"하는 것.
# 비즈니스 로직과 인터페이스를 분리해야 테스트가 쉽다.


def build_index(
    data_path: Path,
    embed_fn,
) -> DocumentStore:
    """문서를 로드하고 임베딩 인덱스를 구축한다.

    Args:
        data_path: JSON 문서 파일 경로.
        embed_fn: 임베딩 생성 함수 (의존성 주입).

    Returns:
        인덱싱된 DocumentStore.
    """
    with data_path.open() as f:
        docs = json.load(f)

    texts = [d["text"] for d in docs]
    embeddings = embed_fn(format_passages(texts))

    store = DocumentStore()
    store.add_documents(texts, embeddings)
    return store


def search_documents(
    query: str,
    store: DocumentStore,
    embed_fn,
    k: int = 5,
) -> list[tuple[str, float]]:
    """쿼리로 문서를 검색한다.

    Args:
        query: 검색 쿼리.
        store: 문서 저장소.
        embed_fn: 임베딩 생성 함수 (의존성 주입).
        k: 반환할 문서 수.

    Returns:
        (문서 텍스트, 유사도 점수) 튜플의 리스트.
    """
    query_embedding = embed_fn(format_queries([query]))
    return retrieve(store, query_embedding, k=k)


# ============================================================
# 3단계: cyclopts CLI 구축
# ============================================================
# cyclopts는 함수의 타입 힌트를 분석하여 자동으로 CLI 인터페이스를 생성한다.
#
# 예시:
#   def hello(name: str, count: int = 1) -> None:
#       for _ in range(count):
#           print(f"Hello, {name}!")
#
# 이 함수를 cyclopts.App에 등록하면:
#   $ my-cli hello --name "World" --count 3


def step3_demo_cli() -> None:
    """CLI 도구의 동작을 시뮬레이션한다."""
    print("=" * 60)
    print("3단계: CLI 도구 동작 데모")
    print("=" * 60)

    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)
    store = build_index(Path("data/sample_docs.json"), embed_fn)

    # CLI에서 실행하는 것과 같은 동작을 함수로 시뮬레이션
    test_queries = [
        "임베딩이란?",
        "RAG가 뭐야?",
        "모델 모니터링 방법",
    ]

    for query in test_queries:
        print(f"\n$ slm search --query '{query}'\n")
        results = search_documents(query, store, embed_fn, k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.4f}] {doc[:70]}...")

    print("\n" + "-" * 60)
    print("실제 CLI 사용법:")
    print("  uv run python -m src.cli.cli_tool search --query '임베딩이란?'")
    print("  uv run python -m src.cli.cli_tool rag --query 'RAG 설명해줘'")
    print("  uv run python -m src.cli.cli_tool embed --text '테스트 문장'")


# ============================================================
# 4단계: CLI 코드 분석
# ============================================================


def step4_code_walkthrough() -> None:
    """src/cli/cli_tool.py의 구조를 설명한다."""
    print("\n" + "=" * 60)
    print("4단계: CLI 코드 구조 분석")
    print("=" * 60)

    print("""
src/cli/cli_tool.py 핵심 구조:

    import cyclopts
    from src.config import get_settings
    from src.embeddings import create_embedding_fn, format_queries

    app = cyclopts.App(name="slm")

    @app.command
    def search(query: str, model: str | None = None) -> None:
        settings = get_settings()                          # 설정 로드
        embed_fn = create_embedding_fn(model or settings.embedding_model_name)  # DI
        results = retrieve(store, embed_fn(format_queries([query])))  # 비즈니스 로직
        for doc, score in results:
            print(f"[{score:.4f}] {doc}")                  # 출력

패턴 요약:
1. 설정은 get_settings()에서 가져온다 (.env 파일 기반)
2. 무거운 의존성(모델)은 함수 내에서 생성한다
3. 비즈니스 로직은 별도 함수로 분리한다
4. CLI 함수는 "조립"만 한다 (의존성 주입)

테스트할 때:
    - build_index()와 search_documents()를 직접 호출
    - embed_fn에 mock을 주입하면 모델 로드 없이 테스트 가능
""")


# ============================================================
# 실행
# ============================================================


def main() -> None:
    """CLI 도구 개발 튜토리얼을 실행한다."""
    step3_demo_cli()
    step4_code_walkthrough()

    print("\n" + "=" * 60)
    print("실습 과제")
    print("=" * 60)
    print("""
1. src/cli/cli_tool.py를 읽고 구조를 파악하세요.
2. 새로운 CLI 커맨드 'stats'를 추가해보세요:
   - 인덱스된 문서 수, 임베딩 차원, 평균 문서 길이를 출력
3. --output json 옵션을 추가하여 JSON 출력을 지원해보세요.
4. embed_fn에 mock 함수를 넣어서 모델 없이 search_documents를
   테스트하는 코드를 작성해보세요.
""")


if __name__ == "__main__":
    main()
