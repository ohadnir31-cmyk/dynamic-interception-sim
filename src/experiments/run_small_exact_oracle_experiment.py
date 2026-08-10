from __future__ import annotations

"""Compare learned and fixed policies with an exact adaptive portfolio oracle
on a small, pre-specified scenario suite.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.experiments.adaptive_portfolio_oracle import (
    OracleSearchLimitExceeded,
    exact_adaptive_portfolio_oracle,
    make_small_oracle_scenarios,
)
from src.experiments.closed_loop_fc_selector import (
    FixedContinuationRegretSelector,
    display_heuristic_name,
    run_closed_loop_selector,
    run_one_shot_selector,
)
from src.sim.runner import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact adaptive heuristic-portfolio enumeration on small cases."
    )
    parser.add_argument("--model-in", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--n-scenarios", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--min-targets", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument("--horizon", type=float, default=8.0)
    parser.add_argument("--max-decisions", type=int, default=6)
    parser.add_argument("--max-search-nodes", type=int, default=250_000)
    parser.add_argument(
        "--allow-zero-oracle-advantage",
        action="store_true",
        help=(
            "Accept nontrivial fixed-policy cases even when the exact adaptive "
            "oracle does not exceed the best fixed heuristic."
        ),
    )
    parser.add_argument("--max-candidate-scenarios", type=int, default=5_000)
    return parser.parse_args()


def _gap_closed(learned: float, baseline: float, oracle: float) -> float:
    denominator = float(oracle - baseline)
    if denominator <= 0:
        return np.nan
    return float((learned - baseline) / denominator)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selector, metadata = FixedContinuationRegretSelector.load(Path(args.model_in))
    scenarios = make_small_oracle_scenarios(
        args.n_scenarios,
        seed=args.seed,
        min_targets=args.min_targets,
        max_targets=args.max_targets,
        horizon=args.horizon,
        require_oracle_advantage=not args.allow_zero_oracle_advantage,
        max_candidate_scenarios=args.max_candidate_scenarios,
    )

    rows: List[Dict[str, Any]] = []
    paths: Dict[str, Any] = {}
    params_rows: List[Dict[str, Any]] = []

    for name, params in tqdm(scenarios.items(), desc="Small exact-oracle cases"):
        params_rows.append({"scenario": name, **asdict(params)})
        fixed_results: Dict[str, Dict[str, Any]] = {}
        for heuristic in selector.candidate_heuristics:
            fixed_results[heuristic] = run_episode(
                params,
                heuristic_name=heuristic,
                preempt=False,
            )

        fixed_best_intercepted = max(
            int(result["intercepted"]) for result in fixed_results.values()
        )
        fixed_winners = sorted(
            display_heuristic_name(code)
            for code, result in fixed_results.items()
            if int(result["intercepted"]) == fixed_best_intercepted
        )
        nt_result = fixed_results.get("NI") or fixed_results.get("NT")
        if nt_result is None:
            raise ValueError("The learned portfolio does not contain NT/NI.")

        learned = run_closed_loop_selector(params, selector, collect_decisions=True)
        one_shot = run_one_shot_selector(params, selector, collect_decisions=True)

        try:
            oracle = exact_adaptive_portfolio_oracle(
                params,
                heuristic_names=selector.candidate_heuristics,
                max_decisions=args.max_decisions,
                max_nodes=args.max_search_nodes,
                require_valid_proposal_from_all_heuristics=True,
            )
            exact = True
            error = ""
        except OracleSearchLimitExceeded as exc:
            oracle = None
            exact = False
            error = str(exc)

        oracle_intercepted = int(oracle.intercepted) if oracle is not None else np.nan
        rows.append(
            {
                "scenario": name,
                "targets": len(params.manual_threats or []),
                "always_nt_intercepted": int(nt_result["intercepted"]),
                "best_fixed_intercepted": int(fixed_best_intercepted),
                "best_fixed_policy_set": ",".join(fixed_winners),
                "one_shot_intercepted": int(one_shot["intercepted"]),
                "learned_closed_loop_intercepted": int(learned["intercepted"]),
                "exact_adaptive_portfolio_intercepted": oracle_intercepted,
                "learned_gap_closed_vs_best_fixed": _gap_closed(
                    learned["intercepted"],
                    fixed_best_intercepted,
                    oracle_intercepted,
                )
                if exact
                else np.nan,
                "learned_gap_closed_vs_nt": _gap_closed(
                    learned["intercepted"],
                    nt_result["intercepted"],
                    oracle_intercepted,
                )
                if exact
                else np.nan,
                "exact": exact,
                "oracle_nodes": int(oracle.stats.nodes) if oracle is not None else np.nan,
                "oracle_leaves": int(oracle.stats.leaves) if oracle is not None else np.nan,
                "oracle_maximum_depth": int(oracle.stats.maximum_depth_reached)
                if oracle is not None
                else np.nan,
                "deduplicated_heuristic_proposals": int(
                    oracle.stats.deduplicated_heuristic_proposals
                )
                if oracle is not None
                else np.nan,
                "error": error,
            }
        )
        paths[name] = {
            "learned_decision_log": learned["decision_log"],
            "one_shot_decision_log": one_shot["decision_log"],
            "oracle_path": oracle.path if oracle is not None else [],
            "oracle_error": error,
        }

    results = pd.DataFrame(rows)
    exact_rows = results[results["exact"]].copy()
    summary = pd.DataFrame(
        [
            {
                "scenarios_requested": int(args.n_scenarios),
                "exact_scenarios": int(len(exact_rows)),
                "mean_always_nt_intercepted": float(
                    exact_rows["always_nt_intercepted"].mean()
                ),
                "mean_best_fixed_intercepted": float(
                    exact_rows["best_fixed_intercepted"].mean()
                ),
                "mean_one_shot_intercepted": float(
                    exact_rows["one_shot_intercepted"].mean()
                ),
                "mean_learned_closed_loop_intercepted": float(
                    exact_rows["learned_closed_loop_intercepted"].mean()
                ),
                "mean_exact_adaptive_portfolio_intercepted": float(
                    exact_rows["exact_adaptive_portfolio_intercepted"].mean()
                ),
                "mean_learned_gap_closed_vs_best_fixed": float(
                    exact_rows["learned_gap_closed_vs_best_fixed"].mean()
                ),
                "mean_learned_gap_closed_vs_nt": float(
                    exact_rows["learned_gap_closed_vs_nt"].mean()
                ),
            }
        ]
    )

    pd.DataFrame(params_rows).to_csv(
        output_dir / "small_exact_scenario_params.csv", index=False
    )
    results.to_csv(output_dir / "small_exact_oracle_results.csv", index=False)
    summary.to_csv(output_dir / "small_exact_oracle_summary.csv", index=False)
    with (output_dir / "small_exact_oracle_paths.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(paths, handle, indent=2, ensure_ascii=False, default=str)
    with (output_dir / "small_exact_oracle_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "oracle_scope": (
                    "exact over distinct target actions proposed by the implemented "
                    "heuristic portfolio at each decision epoch"
                ),
                "unrestricted_target_oracle": False,
                "model_in": args.model_in,
                "model_metadata": metadata,
                "arguments": vars(args),
            },
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("\nSmall exact-oracle results")
    print("--------------------------")
    print(results.to_string(index=False))
    print("\nSummary")
    print("-------")
    print(summary.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
