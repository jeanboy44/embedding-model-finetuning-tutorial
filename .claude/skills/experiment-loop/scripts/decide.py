"""합격선 계산: 대조군 seed들과 후보 run들의 공식 dev 평가로 constraints.md 3절 기준을 계산해 decision.json에 쓴다.

사용:
  uv run python .claude/skills/experiment-loop/scripts/decide.py <실험> \
      --control ../001_baseline_seeds/s42 ../001_baseline_seeds/s43 ../001_baseline_seeds/s44 \
      --candidate A=A_s42,A_s43 B=B_s42 C=C_s42 [--min-delta 0.02]
  (run 경로는 runs/<실험>/ 기준 상대 경로. 후보 하나에 seed run을 쉼표로 여럿)

판정(후보마다):
  1 Δ(후보 seed 평균 − 대조군 seed 평균) ≥ min-delta
  2 Δ > 대조군 seed 범위(max − min)
  3 대응 검정(대조군 첫 seed vs 후보 첫 seed) 95% CI 하한 > 0, McNemar p(후보 사이 Holm 보정) < 0.05
  4 법령 평균 R@5가 표준오차 이상 떨어지지 않음, 법령별 McNemar(Holm)로 유의하게 나빠진 법령 0개
  5 dev-b에서 Δ(seed 평균) > 0
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))
from loop_lib import ROOT, holm, per_law_worse, recall


def load(run_root: Path, rel: str, name: str) -> dict:
    return json.loads((run_root / rel / name).read_text(encoding="utf-8"))


def law_mean(result: dict) -> float:
    laws: dict[str, list[int]] = {}
    for r in result["per_question"]:
        laws.setdefault(r.get("law") or "unknown", []).append(int((r.get("doc_rank") or 10**9) <= 5))
    return mean(mean(v) for v in laws.values())


def paired(base: dict, other: dict) -> dict:
    from ragkit.evaluation import paired_test
    return paired_test(base, other, judge="doc", k=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--control", nargs="+", required=True)
    ap.add_argument("--candidate", nargs="+", required=True, help="이름=run1,run2")
    ap.add_argument("--min-delta", type=float, default=0.02)
    args = ap.parse_args()
    run_root = ROOT / "runs" / args.experiment

    ctrl = [load(run_root, c, "eval_dev.json") for c in args.control]
    ctrl_b = [load(run_root, c, "eval_dev_b.json") for c in args.control]
    c_r5 = [recall(r["per_question"]) for r in ctrl]
    c_mean, c_range = mean(c_r5), max(c_r5) - min(c_r5)
    n = len(ctrl[0]["per_question"])
    se = (c_mean * (1 - c_mean) / n) ** 0.5
    c_law = mean(law_mean(r) for r in ctrl)
    c_b = mean(recall(r["per_question"]) for r in ctrl_b)

    out = {"control": {"runs": args.control, "r5": c_r5, "mean": c_mean, "range": c_range, "se": se,
                       "law_mean": c_law, "dev_b_mean": c_b, "n": n},
           "min_delta": args.min_delta, "candidates": {}}
    raw_p = {}
    for spec in args.candidate:
        name, runs = spec.split("=")
        runs = runs.split(",")
        res = [load(run_root, r, "eval_dev.json") for r in runs]
        res_b = [load(run_root, r, "eval_dev_b.json") for r in runs]
        r5 = [recall(r["per_question"]) for r in res]
        pt = paired(ctrl[0], res[0])
        laws = per_law_worse(ctrl[0], res[0])
        d = mean(r5) - c_mean
        d_b = mean(recall(r["per_question"]) for r in res_b) - c_b
        l_mean = mean(law_mean(r) for r in res)
        raw_p[name] = pt["mcnemar_p"]
        out["candidates"][name] = {
            "runs": runs, "r5": r5, "mean": mean(r5), "delta": d, "delta_dev_b": d_b,
            "paired_vs_control_first_seed": {k: pt[k] for k in ("delta", "ci95", "fixed", "broken", "mcnemar_p")},
            "law_mean": l_mean, "laws_worse_significant": laws["worse"], "n_laws": laws["n_laws"],
            "checks": {
                "1_delta_ge_min": d >= args.min_delta,
                "2_delta_gt_seed_range": d > c_range,
                "3_ci_low_gt_0": pt["ci95"][0] > 0,
                "4_law_mean_not_down": (l_mean - c_law) > -se,
                "4_no_law_significantly_worse": not laws["worse"],
                "5_dev_b_positive": d_b > 0,
                "seeds": len(runs),
            },
        }
    adj = holm(raw_p)
    for name, c in out["candidates"].items():
        c["mcnemar_p_holm"] = adj[name]
        c["checks"]["3_mcnemar_holm_lt_005"] = adj[name] < 0.05
        c["pass"] = all(v for k, v in c["checks"].items() if k != "seeds") and c["checks"]["seeds"] >= 2
        c["pass_except_seeds"] = all(v for k, v in c["checks"].items() if k != "seeds")
    (run_root / "decision.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"대조군 평균 {c_mean:.4f} (범위 {c_range:.4f}, SE {se:.4f}, dev-b {c_b:.4f})")
    for name, c in out["candidates"].items():
        print(f"{name}: 평균 {c['mean']:.4f} Δ {c['delta']:+.4f} dev-b Δ {c['delta_dev_b']:+.4f} "
              f"CI {c['paired_vs_control_first_seed']['ci95']} p(Holm) {c['mcnemar_p_holm']:.2g} "
              f"나빠진 법령 {len(c['laws_worse_significant'])} → {'합격' if c['pass'] else ('seed 확인 필요' if c['pass_except_seeds'] else '불합격')}")


if __name__ == "__main__":
    main()
