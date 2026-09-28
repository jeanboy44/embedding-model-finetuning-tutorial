"""
Phase 4-1: 로깅과 관찰성
===========================

학습 목표:
- 왜 ML 시스템에 로깅이 필요한지 이해한다
- loguru로 구조화된 로깅을 설정한다
- 검색/생성 파이프라인에 로깅을 삽입한다
- 로그를 분석하여 문제를 진단한다

실행:
    uv run python tutorials/_legacy/monitoring/01_logging.py
"""

import json
import time
from pathlib import Path

from loguru import logger

from ragkit.config import get_settings
from ragkit.embeddings import create_embedding_fn, format_passages, format_queries
from ragkit.monitoring.logger import (
    log_latency,
    log_query,
    log_retrieval_results,
)
from ragkit.retrieval import DocumentStore, retrieve

# ============================================================
# 1단계: 왜 로깅이 필요한가?
# ============================================================


def step1_why_logging() -> None:
    """ML 시스템에서 로깅이 중요한 이유를 설명한다."""
    print("=" * 60)
    print("1단계: 왜 로깅이 필요한가?")
    print("=" * 60)

    print("""
ML 시스템은 전통적인 소프트웨어와 다르게 "조용히 실패"한다.

일반 소프트웨어: 버그 → 에러 → 크래시 (명확)
ML 시스템:      성능 저하 → 결과 품질 하락 → 사용자 불만 (불명확)

로깅이 필요한 상황:
1. "어제까지 잘 되던 검색이 오늘 이상해요"
   → 임베딩 모델 업데이트? 데이터 변경? 로그로 확인

2. "특정 쿼리에서 답변이 이상해요"
   → 어떤 문서가 검색됐는지? 유사도 점수는? 프롬프트는?

3. "시스템이 느려졌어요"
   → 어느 단계에서 병목? 임베딩 생성? 검색? LLM 호출?

4. "모델을 업데이트했는데 성능이 올랐는지 모르겠어요"
   → 이전/이후 메트릭 비교
""")


# ============================================================
# 2단계: loguru 기초
# ============================================================


def step2_loguru_basics() -> None:
    """loguru의 기본 사용법을 실습한다."""
    print("\n" + "=" * 60)
    print("2단계: loguru 기초")
    print("=" * 60)

    print("\nloguru는 Python의 표준 logging보다 훨씬 간편하다:")
    print("  - import 하나로 바로 사용")
    print("  - 컬러 출력, 자동 포맷팅")
    print("  - 파일 회전, 구조화 로깅 지원\n")

    # 기본 로깅
    logger.info("기본 정보 메시지")
    logger.warning("경고 메시지")
    logger.error("에러 메시지")

    # 구조화 로깅 (변수 포함)
    query = "임베딩이란?"
    k = 3
    latency = 45.2
    logger.info(f"검색 완료: query='{query}' k={k} latency={latency:.1f}ms")

    # 파일로 저장
    log_path = Path("logs/tutorial.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 파일 핸들러 추가 (JSON 형식)
    handler_id = logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        rotation="10 MB",
        retention="7 days",
    )

    logger.info("이 메시지는 파일에도 기록됩니다")
    logger.info(f"로그 파일: {log_path}")

    # 핸들러 제거 (다른 단계에 영향 안 주도록)
    logger.remove(handler_id)


# ============================================================
# 3단계: 파이프라인에 로깅 삽입
# ============================================================


def step3_pipeline_logging() -> None:
    """검색 파이프라인에 로깅을 삽입하여 실행한다."""
    print("\n" + "=" * 60)
    print("3단계: 파이프라인에 로깅 삽입")
    print("=" * 60)

    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)

    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)
    embeddings = embed_fn(format_passages([d["text"] for d in docs]))

    store = DocumentStore()
    store.add_documents([d["text"] for d in docs], embeddings)

    queries = [
        "임베딩이란 무엇인가?",
        "RAG 시스템의 장단점",
        "어떤 모니터링 도구를 써야 하나?",
    ]

    print("\n파이프라인 실행 (로그 포함):\n")

    for query in queries:
        # 1. 쿼리 로깅
        start = time.time()
        query_emb = embed_fn(format_queries([query]))
        embed_time = (time.time() - start) * 1000
        log_query(query, query_emb.shape)
        log_latency("embedding", embed_time)

        # 2. 검색 + 결과 로깅
        start = time.time()
        results = retrieve(store, query_emb, k=3)
        search_time = (time.time() - start) * 1000
        scores = [s for _, s in results]
        log_retrieval_results(query, len(results), scores)
        log_latency("retrieval", search_time)

        # 3. 전체 지연시간
        total = embed_time + search_time
        log_latency("total", total)

        print(f"  [{total:.1f}ms] '{query[:30]}...' → top score: {scores[0]:.4f}")


# ============================================================
# 4단계: 로그 분석
# ============================================================


def step4_log_analysis() -> None:
    """로그 데이터를 분석하여 인사이트를 도출한다."""
    print("\n" + "=" * 60)
    print("4단계: 로그 분석 패턴")
    print("=" * 60)

    print("""
로그에서 확인해야 할 것:

1. 지연시간 분포
   - 평균/P50/P95/P99 확인
   - 특정 쿼리에서 급증하는 패턴?

2. 검색 품질 트렌드
   - 유사도 점수의 평균이 하락하고 있는가?
   - 특정 카테고리에서 점수가 낮은가?

3. 에러 패턴
   - 반복적으로 실패하는 쿼리가 있는가?
   - 특정 시간대에 에러가 집중되는가?

4. 사용 패턴
   - 가장 많이 검색되는 주제는?
   - 사용자가 자주 재시도하는 쿼리는? (불만족 신호)
""")

    # 시뮬레이션: 로그 데이터에서 인사이트 추출
    print("시뮬레이션: 지연시간 분석\n")

    import random

    random.seed(42)

    # 가상 로그 데이터 생성
    latencies = [random.gauss(50, 15) for _ in range(100)]
    # 일부 느린 쿼리 추가
    latencies.extend([random.gauss(200, 50) for _ in range(5)])

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

    print(f"  총 쿼리 수: {len(latencies)}")
    print(f"  평균: {sum(latencies) / len(latencies):.1f}ms")
    print(f"  P50:  {p50:.1f}ms")
    print(f"  P95:  {p95:.1f}ms")
    print(f"  P99:  {p99:.1f}ms")

    slow = sum(1 for l in latencies if l > 100)
    print(f"\n  100ms 초과 쿼리: {slow}개 ({slow / len(latencies) * 100:.1f}%)")
    print("  → P95와 P99 사이 큰 갭 = 일부 느린 쿼리 존재")
    print("  → 원인 분석 필요 (긴 쿼리? 캐시 미스? 모델 로드?)")


# ============================================================
# 실행
# ============================================================


def main() -> None:
    """로깅과 관찰성 튜토리얼을 실행한다."""
    step1_why_logging()
    step2_loguru_basics()
    step3_pipeline_logging()
    step4_log_analysis()

    print("\n" + "=" * 60)
    print("실습 과제")
    print("=" * 60)
    print("""
1. logs/tutorial.log 파일을 열어 기록된 내용을 확인하세요.
2. loguru의 JSON 직렬화 기능을 사용하여 구조화 로그를 출력해보세요:
   logger.add("logs/structured.log", serialize=True)
3. 검색 결과의 유사도 점수가 0.5 미만이면 warning을 로깅하도록 수정하세요.
4. src/monitoring/logger.py에 새로운 로깅 함수를 추가해보세요.
""")


if __name__ == "__main__":
    main()
