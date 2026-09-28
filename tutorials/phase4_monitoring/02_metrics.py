"""
Phase 4-2: 메트릭 수집과 분석
================================

학습 목표:
- ML 시스템의 핵심 메트릭을 정의한다
- MetricsCollector로 메트릭을 수집하고 저장한다
- 수집된 메트릭을 분석하여 리포트를 생성한다
- 알림 기준(threshold)을 설정한다

실행:
    uv run python tutorials/phase4_monitoring/02_metrics.py
"""

import json
import random
import time
from pathlib import Path

from src.config import get_settings
from src.embeddings import create_embedding_fn, format_passages, format_queries
from src.monitoring.metrics import MetricsCollector
from src.retrieval import DocumentStore, retrieve

# ============================================================
# 1단계: ML 시스템의 핵심 메트릭
# ============================================================


def step1_key_metrics() -> None:
    """ML 시스템에서 추적해야 할 핵심 메트릭을 설명한다."""
    print("=" * 60)
    print("1단계: ML 시스템의 핵심 메트릭")
    print("=" * 60)

    print("""
RAG 시스템의 4가지 메트릭 카테고리:

1. 검색 품질 (Retrieval Quality)
   - Precision@K: 상위 K개 중 관련 문서 비율
   - 평균 유사도 점수: 검색 결과의 관련성
   - 빈 결과 비율: 아무것도 못 찾은 비율

2. 생성 품질 (Generation Quality)
   - 답변 길이: 너무 짧거나 긴 답변 감지
   - 환각 비율: 검색 문서에 없는 내용 생성
   - 사용자 만족도: 피드백 수집

3. 성능 (Performance)
   - 임베딩 생성 지연시간
   - 검색 지연시간
   - LLM 응답 지연시간
   - 전체 E2E 지연시간

4. 시스템 건강 (System Health)
   - 에러율
   - 처리량 (QPS)
   - 인덱스 크기
   - 메모리 사용량
""")


# ============================================================
# 2단계: MetricsCollector 실습
# ============================================================


def step2_collector_usage() -> MetricsCollector:
    """MetricsCollector를 사용하여 메트릭을 수집한다.

    Returns:
        메트릭이 수집된 MetricsCollector.
    """
    print("\n" + "=" * 60)
    print("2단계: MetricsCollector 실습")
    print("=" * 60)

    # 데이터 준비
    data_path = Path("data/sample_docs.json")
    with data_path.open() as f:
        docs = json.load(f)

    settings = get_settings()
    embed_fn = create_embedding_fn(settings.embedding_model_name)
    embeddings = embed_fn(format_passages([d["text"] for d in docs]))

    store = DocumentStore()
    store.add_documents([d["text"] for d in docs], embeddings)

    # MetricsCollector 생성
    collector = MetricsCollector(output_dir=Path("logs/metrics"))

    # 여러 쿼리 실행하며 메트릭 수집
    queries = [
        "머신러닝이란?",
        "딥러닝과 머신러닝의 차이",
        "임베딩 모델 파인튜닝 방법",
        "RAG 시스템 구축",
        "프롬프트 엔지니어링 팁",
        "모델 성능 모니터링",
        "벡터 검색 방법",
        "트랜스포머 아키텍처",
    ]

    print(f"\n{len(queries)}개 쿼리에 대해 메트릭 수집 중...\n")

    for query in queries:
        # 검색 실행 + 시간 측정
        start = time.time()
        query_emb = embed_fn(format_queries([query]))
        results = retrieve(store, query_emb, k=3)
        latency_ms = (time.time() - start) * 1000

        scores = [s for _, s in results]

        # 메트릭 기록
        collector.log_retrieval(
            query=query,
            num_results=len(results),
            top_scores=scores,
            latency_ms=latency_ms,
        )

        print(f"  [{latency_ms:6.1f}ms] '{query:<25}' top={scores[0]:.4f}")

    # 요약 출력
    summary = collector.get_summary()
    print("\n요약:")
    print(f"  검색 횟수: {summary['retrieval_count']}")
    print(f"  평균 지연시간: {summary['avg_retrieval_latency_ms']:.1f}ms")

    # 저장
    collector.save()
    print("\n  메트릭이 logs/metrics/에 저장되었습니다.")

    return collector


# ============================================================
# 3단계: 메트릭 분석 리포트
# ============================================================


def step3_analysis_report(collector: MetricsCollector) -> None:
    """수집된 메트릭을 분석하여 리포트를 생성한다.

    Args:
        collector: 메트릭이 수집된 MetricsCollector.
    """
    print("\n" + "=" * 60)
    print("3단계: 메트릭 분석 리포트")
    print("=" * 60)

    metrics = collector.retrieval_metrics

    # 지연시간 분석
    latencies = [m.latency_ms for m in metrics]
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]

    print("\n[지연시간 분석]")
    print(f"  평균: {sum(latencies) / len(latencies):.1f}ms")
    print(f"  P50:  {p50:.1f}ms")
    print(f"  P95:  {p95:.1f}ms")
    print(f"  최소: {min(latencies):.1f}ms")
    print(f"  최대: {max(latencies):.1f}ms")

    # 검색 품질 분석
    all_top_scores = [m.top_scores[0] for m in metrics if m.top_scores]
    avg_top_score = sum(all_top_scores) / len(all_top_scores)
    low_quality = sum(1 for s in all_top_scores if s < 0.5)

    print("\n[검색 품질 분석]")
    print(f"  평균 top-1 유사도: {avg_top_score:.4f}")
    print(
        f"  유사도 0.5 미만: {low_quality}/{len(all_top_scores)} ({low_quality / len(all_top_scores) * 100:.0f}%)"
    )

    # 쿼리별 상세
    print("\n[쿼리별 상세]")
    print(f"  {'쿼리':<28} {'지연시간':>10} {'Top Score':>10}")
    print(f"  {'-' * 50}")
    for m in metrics:
        top = m.top_scores[0] if m.top_scores else 0
        flag = " ⚠" if top < 0.5 else ""
        print(f"  {m.query:<28} {m.latency_ms:>8.1f}ms {top:>10.4f}{flag}")


# ============================================================
# 4단계: 알림 기준 설정
# ============================================================


def step4_alerting() -> None:
    """메트릭 기반 알림 규칙을 정의하고 시뮬레이션한다."""
    print("\n" + "=" * 60)
    print("4단계: 알림 기준 설정")
    print("=" * 60)

    print("""
알림 규칙 예시:

  CRITICAL:
    - 에러율 > 5%
    - P95 지연시간 > 5초
    - 서비스 다운

  WARNING:
    - 평균 유사도 점수 < 0.5 (검색 품질 저하)
    - P95 지연시간 > 2초
    - 빈 결과 비율 > 10%

  INFO:
    - 일일 쿼리 수 리포트
    - 주간 성능 트렌드
""")

    # 시뮬레이션: 시간별 메트릭 변화
    print("시뮬레이션: 24시간 메트릭 트렌드\n")

    random.seed(42)

    print(f"  {'시간':>4}  {'QPS':>5}  {'P50(ms)':>8}  {'Avg Score':>10}  상태")
    print(f"  {'-' * 50}")

    for hour in range(24):
        # 시뮬레이션 데이터
        qps = random.randint(5, 50)
        base_latency = 45 if 9 <= hour <= 18 else 30
        p50 = base_latency + random.gauss(0, 10)
        avg_score = 0.72 + random.gauss(0, 0.05)

        # 이상 상황 주입
        if hour == 14:  # 오후 2시에 갑자기 느려짐
            p50 = 200 + random.gauss(0, 20)

        if hour == 20:  # 저녁 8시에 검색 품질 저하
            avg_score = 0.35 + random.gauss(0, 0.05)

        # 알림 판단
        status = "OK"
        if p50 > 100:
            status = "⚠ WARN (지연시간)"
        if avg_score < 0.5:
            status = "⚠ WARN (검색품질)"
        if p50 > 200:
            status = "🔴 CRIT (지연시간)"

        print(f"  {hour:02d}:00  {qps:>5}  {p50:>8.1f}  {avg_score:>10.4f}  {status}")

    print("""
분석:
- 14시: P50 지연시간 급증 → 인프라 문제? 트래픽 급증?
- 20시: 검색 품질 저하 → 데이터 변경? 모델 문제?

실무에서는 Prometheus + Grafana, Datadog 등으로
실시간 대시보드와 알림을 구성합니다.
""")


# ============================================================
# 실행
# ============================================================


def main() -> None:
    """메트릭 수집과 분석 튜토리얼을 실행한다."""
    step1_key_metrics()
    collector = step2_collector_usage()
    step3_analysis_report(collector)
    step4_alerting()

    print("=" * 60)
    print("실습 과제")
    print("=" * 60)
    print("""
1. logs/metrics/ 디렉토리의 JSON 파일을 열어 구조를 파악하세요.
2. MetricsCollector에 log_generation 메서드를 호출하는 코드를 추가하세요.
3. 알림 규칙을 함수로 구현해보세요:
   def check_alerts(collector: MetricsCollector) -> list[str]:
4. (심화) 수집된 메트릭으로 시계열 차트를 그려보세요 (matplotlib 사용).
5. (심화) 검색 품질이 임계값 아래로 떨어지면 자동으로
   슬랙/이메일 알림을 보내는 시스템을 설계해보세요.
""")


if __name__ == "__main__":
    main()
