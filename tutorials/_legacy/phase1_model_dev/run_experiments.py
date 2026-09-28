"""Phase 1: 모델 개발 — 임베딩 방식 비교 실험 실행.

단계별 학습은 01_exploration.py, 02_comparison.py를 참고하세요.
이 스크립트는 experiments/ 폴더의 실험을 직접 실행합니다.
"""

from pathlib import Path

from experiments.exp_001_base_embedding.run import BaseEmbeddingExperiment
from experiments.exp_002_finetuned_embedding.run import FinetuneEmbeddingExperiment
from experiments.exp_003_llm_query_expansion.run import LLMQueryExpansionExperiment


def main() -> None:
    """세 가지 임베딩 실험을 순차 실행하고 결과를 출력한다."""
    exp_dir = Path(__file__).parent.parent.parent / "experiments"

    print("=" * 60)
    print("Phase 1: 모델 개발 - 임베딩 방식 비교")
    print("=" * 60)

    experiments = [
        (
            "Base 임베딩",
            BaseEmbeddingExperiment,
            exp_dir / "exp_001_base_embedding" / "config.yaml",
        ),
        (
            "파인튜닝 임베딩",
            FinetuneEmbeddingExperiment,
            exp_dir / "exp_002_finetuned_embedding" / "config.yaml",
        ),
        (
            "LLM 쿼리 확장",
            LLMQueryExpansionExperiment,
            exp_dir / "exp_003_llm_query_expansion" / "config.yaml",
        ),
    ]

    all_results: dict[str, dict] = {}

    for exp_name, exp_class, config_path in experiments:
        print(f"\n[실행 중: {exp_name}...]")
        try:
            exp = exp_class(config_path)
            results = exp.run()
            all_results[exp_name] = results
            print(f"  {exp_name} 완료됨")
        except (ImportError, FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"  {exp_name} 실패: {e}")

    print("\n" + "=" * 60)
    print("실험 요약")
    print("=" * 60)

    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        if "similarity_scores" in results:
            print(f"  상위 3개 점수: {results['similarity_scores']}")
        top = results.get("top_3_results", [None])
        print(f"  최고 결과: {top[0] if top else 'N/A'}")


if __name__ == "__main__":
    main()
