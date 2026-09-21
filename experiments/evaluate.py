"""모든 실험 실행 및 결과 비교."""

import json
from pathlib import Path

from experiments.exp_001_base_embedding.run import BaseEmbeddingExperiment
from experiments.exp_002_finetuned_embedding.run import FinetuneEmbeddingExperiment
from experiments.exp_003_llm_query_expansion.run import LLMQueryExpansionExperiment


def run_all_experiments() -> None:
    """세 가지 실험을 모두 실행하고 결과를 비교한다."""
    exp_dir = Path(__file__).parent

    experiments = [
        (BaseEmbeddingExperiment, exp_dir / "exp_001_base_embedding" / "config.yaml"),
        (
            FinetuneEmbeddingExperiment,
            exp_dir / "exp_002_finetuned_embedding" / "config.yaml",
        ),
        (
            LLMQueryExpansionExperiment,
            exp_dir / "exp_003_llm_query_expansion" / "config.yaml",
        ),
    ]

    all_results: dict[str, dict] = {}

    for exp_class, config_path in experiments:
        if config_path.exists():
            print(f"\nRunning {exp_class.__name__}...")
            exp = exp_class(config_path)
            results = exp.run()
            all_results[exp.name] = results
            print(f"  {exp.name} completed")
        else:
            print(f"  Config not found: {config_path}")

    comparison_file = exp_dir / "results" / "comparison.json"
    comparison_file.parent.mkdir(parents=True, exist_ok=True)

    with comparison_file.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Comparison saved to {comparison_file}")

    print("\n=== Experiment Comparison ===")
    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        for key, value in results.items():
            if not isinstance(value, list):
                print(f"  {key}: {value}")


if __name__ == "__main__":
    run_all_experiments()
