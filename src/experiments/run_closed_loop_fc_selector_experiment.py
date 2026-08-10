from __future__ import annotations

"""Train the existing fixed-continuation selector and test it as a true
closed-loop adaptive switching policy on new scenarios.

No Always-NT bootstrap and no new adaptive-label training are performed.  The
existing rollout dataset supplies the supervised labels for ``mu_FC``.  Always
NT and the other fixed heuristics appear only as evaluation baselines.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.experiments.closed_loop_fc_selector import (
    DEFAULT_CANDIDATE_HEURISTICS,
    FixedContinuationRegretSelector,
    display_heuristic_name,
    run_closed_loop_selector,
    train_mu_fc_from_existing_dataset,
)
from src.experiments.run_large_scale_rollout import make_large_scale_scenarios
from src.sim.runner import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train mu_FC from the existing fixed-continuation rollout dataset "
            "and evaluate it as a closed-loop heuristic-switching policy."
        )
    )
    parser.add_argument("--output-dir", required=True, type=str)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--rollout-input-dir",
        type=str,
        help="Directory containing large_scale_rollout_states*.csv.",
    )
    source.add_argument(
        "--model-in",
        type=str,
        help="Previously saved mu_FC model bundle; skips retraining.",
    )

    parser.add_argument(
        "--dataset-mode",
        choices=["no_ties", "with_ties", "full"],
        default="no_ties",
    )
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "margin", "oracle_gap"],
        default="margin",
    )
    parser.add_argument("--weight-alpha", type=float, default=0.25)
    parser.add_argument("--clip-abs", type=float, default=1_000_000.0)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument(
        "--regret-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum predicted-regret improvement required before leaving the "
            "threshold baseline. The unit is predicted future interceptions."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["none", "nt_override", "previous"],
        default="nt_override",
        help=(
            "'nt_override' keeps NT unless an alternative beats it by the "
            "threshold; 'previous' retains the preceding heuristic unless the "
            "new best is better by the threshold; 'none' disables gating."
        ),
    )
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="",
        help=(
            "Optional comma-separated threshold grid, e.g. "
            "'0,0.25,0.5,1,1.5,2,3'. The threshold is selected on a separate "
            "closed-loop validation set."
        ),
    )
    parser.add_argument("--n-threshold-validation-scenarios", type=int, default=0)
    parser.add_argument("--threshold-validation-seed", type=int, default=20260809)

    parser.add_argument("--n-test-scenarios", type=int, default=250)
    parser.add_argument("--test-seed", type=int, default=20260810)
    parser.add_argument(
        "--scenario-mix",
        choices=["baseline", "decision_rich", "heavy_load"],
        default="decision_rich",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=731)
    parser.add_argument(
        "--no-decision-log",
        action="store_true",
        help="Do not save the per-decision adaptive trace.",
    )
    return parser.parse_args()


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_samples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    means = np.empty(int(n_samples), dtype=float)
    for i in range(int(n_samples)):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = float(np.mean(sample))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _policy_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for policy, group in results.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "scenarios": int(len(group)),
                "mean_intercepted": float(group["intercepted"].mean()),
                "std_intercepted": float(group["intercepted"].std()),
                "median_intercepted": float(group["intercepted"].median()),
                "mean_escaped": float(group["escaped"].mean()),
                "mean_spawned": float(group["spawned"].mean()),
                "mean_active_at_horizon": float(
                    (group["spawned"] - group["intercepted"] - group["escaped"]).mean()
                ),
                "mean_interception_rate": float(group["interception_rate"].mean()),
                "mean_decisions": float(group["num_decisions"].mean()),
                "mean_switches": float(group["num_heuristic_switches"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_intercepted", "mean_escaped"],
        ascending=[False, True],
    )


def _paired_comparison_to_reference(
    results: pd.DataFrame,
    reference_policy: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    pivot_i = results.pivot(index="scenario", columns="policy", values="intercepted")
    pivot_e = results.pivot(index="scenario", columns="policy", values="escaped")
    if reference_policy not in pivot_i.columns:
        raise ValueError(f"Reference policy not found: {reference_policy}")

    rows: List[Dict[str, Any]] = []
    for offset, policy in enumerate(pivot_i.columns):
        if policy == reference_policy:
            continue
        valid = pivot_i[[policy, reference_policy]].dropna()
        difference = (
            valid[policy].to_numpy(dtype=float)
            - valid[reference_policy].to_numpy(dtype=float)
        )
        escaped_valid = pivot_e[[policy, reference_policy]].dropna()
        escaped_difference = (
            escaped_valid[policy].to_numpy(dtype=float)
            - escaped_valid[reference_policy].to_numpy(dtype=float)
        )
        ci_low, ci_high = _bootstrap_mean_ci(
            difference,
            n_samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
        )
        rows.append(
            {
                "policy": policy,
                "reference_policy": reference_policy,
                "scenarios": int(len(difference)),
                "mean_intercepted_difference": float(np.mean(difference)),
                "median_intercepted_difference": float(np.median(difference)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "win_rate": float(np.mean(difference > 0)),
                "tie_rate": float(np.mean(difference == 0)),
                "loss_rate": float(np.mean(difference < 0)),
                "mean_escaped_difference": float(np.mean(escaped_difference)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_intercepted_difference",
        ascending=False,
    )


def _best_fixed_hindsight(results: pd.DataFrame, fixed_policies: Sequence[str]) -> pd.DataFrame:
    fixed = results[results["policy"].isin(fixed_policies)].copy()
    rows: List[Dict[str, Any]] = []
    for scenario, group in fixed.groupby("scenario"):
        best_intercepted = int(group["intercepted"].max())
        candidates = group[group["intercepted"] == best_intercepted]
        best_escaped = int(candidates["escaped"].min())
        winners = candidates[candidates["escaped"] == best_escaped]["policy"].tolist()
        rows.append(
            {
                "scenario": scenario,
                "best_fixed_intercepted": best_intercepted,
                "best_fixed_escaped": best_escaped,
                "best_fixed_policy_set": ",".join(sorted(winners)),
            }
        )
    return pd.DataFrame(rows)


def _heuristic_usage(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(
            columns=["heuristic", "decisions", "decision_share", "scenarios_used"]
        )
    counts = (
        decisions.groupby("heuristic")
        .agg(
            decisions=("heuristic", "size"),
            scenarios_used=("scenario", "nunique"),
        )
        .reset_index()
    )
    counts["decision_share"] = counts["decisions"] / max(1, counts["decisions"].sum())
    return counts.sort_values("decisions", ascending=False)


def _write_training_outputs(artifacts: Any, output_dir: Path) -> None:
    artifacts.validation_summary.to_csv(
        output_dir / "mu_fc_internal_validation_summary.csv",
        index=False,
    )
    artifacts.validation_by_bucket.to_csv(
        output_dir / "mu_fc_internal_validation_by_bucket.csv",
        index=False,
    )
    artifacts.validation_model_fit.to_csv(
        output_dir / "mu_fc_internal_validation_model_fit.csv",
        index=False,
    )
    artifacts.validation_predictions.to_csv(
        output_dir / "mu_fc_internal_validation_predictions.csv",
        index=False,
    )
    artifacts.validation_cleaning_report.to_csv(
        output_dir / "mu_fc_validation_feature_cleaning.csv",
        index=False,
    )
    artifacts.full_cleaning_report.to_csv(
        output_dir / "mu_fc_full_fit_feature_cleaning.csv",
        index=False,
    )



def _parse_threshold_grid(value: str) -> List[float]:
    if not value.strip():
        return []
    return sorted(
        {
            max(0.0, float(token.strip()))
            for token in value.split(",")
            if token.strip()
        }
    )


def _selector_with_threshold(
    selector: FixedContinuationRegretSelector,
    *,
    threshold: float,
    threshold_mode: str,
    name: Optional[str] = None,
) -> FixedContinuationRegretSelector:
    return FixedContinuationRegretSelector(
        models=selector.models,
        feature_columns=selector.feature_columns,
        medians=selector.medians,
        candidate_heuristics=selector.candidate_heuristics,
        clip_abs=selector.clip_abs,
        regret_threshold=float(threshold),
        threshold_mode=threshold_mode,
        name=name
        or (
            f"Adaptive mu_FC selector (mode={threshold_mode}, "
            f"tau={float(threshold):g})"
        ),
    )


def _tune_regret_threshold(
    selector: FixedContinuationRegretSelector,
    *,
    thresholds: Sequence[float],
    n_scenarios: int,
    seed: int,
    scenario_mix: str,
    threshold_mode: str,
    output_dir: Path,
) -> float:
    """Choose the threshold on a fresh closed-loop validation scenario set."""
    if not thresholds:
        return float(selector.regret_threshold)
    if n_scenarios <= 0:
        raise ValueError(
            "A threshold grid was supplied, but "
            "--n-threshold-validation-scenarios is not positive."
        )

    scenarios = make_large_scale_scenarios(
        n_scenarios=int(n_scenarios),
        seed=int(seed),
        scenario_mix=scenario_mix,
    )

    nt_by_scenario: Dict[str, int] = {}
    for scenario_name, scenario_obj in tqdm(
        scenarios.items(),
        desc="Threshold validation: Always NT",
    ):
        fixed = run_episode(scenario_obj.params, heuristic_name="NI", preempt=False)
        nt_by_scenario[scenario_name] = int(fixed["intercepted"])

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        candidate = _selector_with_threshold(
            selector,
            threshold=float(threshold),
            threshold_mode=threshold_mode,
        )
        intercepted_values: List[int] = []
        escaped_values: List[int] = []
        active_values: List[int] = []
        switch_values: List[int] = []
        differences: List[int] = []

        for scenario_name, scenario_obj in tqdm(
            scenarios.items(),
            desc=f"Threshold validation tau={threshold:g}",
            leave=False,
        ):
            result = run_closed_loop_selector(
                scenario_obj.params,
                candidate,
                collect_decisions=False,
            )
            intercepted = int(result["intercepted"])
            escaped = int(result["escaped"])
            active = int(result["spawned"] - intercepted - escaped)
            intercepted_values.append(intercepted)
            escaped_values.append(escaped)
            active_values.append(active)
            switch_values.append(int(result["num_heuristic_switches"]))
            differences.append(intercepted - nt_by_scenario[scenario_name])

        diff_array = np.asarray(differences)
        rows.append(
            {
                "threshold": float(threshold),
                "threshold_mode": threshold_mode,
                "scenarios": int(n_scenarios),
                "mean_intercepted": float(np.mean(intercepted_values)),
                "mean_difference_vs_NT": float(np.mean(diff_array)),
                "win_rate_vs_NT": float(np.mean(diff_array > 0)),
                "tie_rate_vs_NT": float(np.mean(diff_array == 0)),
                "loss_rate_vs_NT": float(np.mean(diff_array < 0)),
                "mean_escaped": float(np.mean(escaped_values)),
                "mean_active_at_horizon": float(np.mean(active_values)),
                "mean_switches": float(np.mean(switch_values)),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "closed_loop_threshold_validation.csv", index=False)
    ranked = summary.sort_values(
        [
            "mean_intercepted",
            "mean_escaped",
            "mean_active_at_horizon",
            "mean_switches",
            "threshold",
        ],
        ascending=[False, True, True, True, False],
    )
    chosen = float(ranked.iloc[0]["threshold"])
    print("\nThreshold validation summary")
    print("----------------------------")
    print(summary.to_string(index=False))
    print(f"Selected regret threshold: {chosen:g}")
    return chosen

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Closed-loop evaluation of mu_FC ===")
    print(
        "Training labels: existing fixed-continuation rollout data; "
        "deployment: fresh heuristic selection after each pursued-target resolution."
    )

    if args.model_in:
        selector, training_metadata = FixedContinuationRegretSelector.load(
            Path(args.model_in)
        )
        model_path = Path(args.model_in)
        print(f"Loaded selector: {model_path}")
    else:
        artifacts = train_mu_fc_from_existing_dataset(
            Path(args.rollout_input_dir),
            dataset_mode=args.dataset_mode,
            validation_size=args.validation_size,
            random_state=args.random_state,
            sample_weight_mode=args.sample_weight_mode,
            weight_alpha=args.weight_alpha,
            clip_abs=args.clip_abs,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=args.max_depth,
        )
        selector = artifacts.selector
        training_metadata = artifacts.metadata
        _write_training_outputs(artifacts, output_dir)
        model_path = output_dir / "mu_fc_selector.joblib"
        selector.save(model_path, training_metadata)
        print(f"Saved selector: {model_path}")
        print("Internal validation:")
        print(artifacts.validation_summary.to_string(index=False))

    thresholds = _parse_threshold_grid(args.threshold_grid)
    selected_threshold = float(args.regret_threshold)
    if thresholds:
        selected_threshold = _tune_regret_threshold(
            selector,
            thresholds=thresholds,
            n_scenarios=args.n_threshold_validation_scenarios,
            seed=args.threshold_validation_seed,
            scenario_mix=args.scenario_mix,
            threshold_mode=args.threshold_mode,
            output_dir=output_dir,
        )

    selector = _selector_with_threshold(
        selector,
        threshold=selected_threshold,
        threshold_mode=args.threshold_mode,
    )
    configured_model_path = output_dir / "mu_fc_selector_configured.joblib"
    selector.save(
        configured_model_path,
        {
            **training_metadata,
            "regret_threshold": selected_threshold,
            "threshold_mode": args.threshold_mode,
        },
    )
    print(
        f"Closed-loop gate: mode={args.threshold_mode}, "
        f"threshold={selected_threshold:g}"
    )

    scenarios = make_large_scale_scenarios(
        n_scenarios=args.n_test_scenarios,
        seed=args.test_seed,
        scenario_mix=args.scenario_mix,
    )

    scenario_params = pd.DataFrame(
        [
            {
                "scenario": f"testseed_{args.test_seed}_{name}",
                "scenario_mix": args.scenario_mix,
                **asdict(obj.params),
            }
            for name, obj in scenarios.items()
        ]
    )
    scenario_params.to_csv(output_dir / "closed_loop_test_scenario_params.csv", index=False)

    fixed_heuristics = list(selector.candidate_heuristics)
    fixed_policy_names = [f"Always {display_heuristic_name(h)}" for h in fixed_heuristics]
    adaptive_policy_name = selector.name

    result_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []

    for scenario_name, scenario_obj in tqdm(
        scenarios.items(),
        desc="Closed-loop test scenarios",
    ):
        output_scenario_name = f"testseed_{args.test_seed}_{scenario_name}"
        params = scenario_obj.params

        adaptive = run_closed_loop_selector(
            params,
            selector,
            collect_decisions=not args.no_decision_log,
        )
        result_rows.append(
            {
                "scenario": output_scenario_name,
                "policy": adaptive_policy_name,
                "policy_type": "adaptive_closed_loop_mu_FC",
                "scenario_regime": params.scenario_regime,
                "spatial_structure": params.spatial_structure,
                "arrival_process": params.arrival_process,
                "deadline_pressure": params.deadline_pressure,
                "spawned": adaptive["spawned"],
                "intercepted": adaptive["intercepted"],
                "escaped": adaptive["escaped"],
                "interception_rate": adaptive["interception_rate"],
                "num_decisions": adaptive["num_decisions"],
                "num_heuristic_switches": adaptive["num_heuristic_switches"],
            }
        )
        for row in adaptive["decision_log"]:
            decision_rows.append(
                {
                    "scenario": output_scenario_name,
                    "scenario_regime": params.scenario_regime,
                    "spatial_structure": params.spatial_structure,
                    "arrival_process": params.arrival_process,
                    "deadline_pressure": params.deadline_pressure,
                    **row,
                }
            )

        for heuristic in fixed_heuristics:
            fixed = run_episode(params, heuristic_name=heuristic, preempt=False)
            result_rows.append(
                {
                    "scenario": output_scenario_name,
                    "policy": f"Always {display_heuristic_name(heuristic)}",
                    "policy_type": "fixed_heuristic",
                    "scenario_regime": params.scenario_regime,
                    "spatial_structure": params.spatial_structure,
                    "arrival_process": params.arrival_process,
                    "deadline_pressure": params.deadline_pressure,
                    "spawned": int(fixed["spawned"]),
                    "intercepted": int(fixed["intercepted"]),
                    "escaped": int(fixed["escaped"]),
                    "interception_rate": float(
                        fixed["intercepted"] / max(1, fixed["spawned"])
                    ),
                    "num_decisions": np.nan,
                    "num_heuristic_switches": 0,
                }
            )

    results = pd.DataFrame(result_rows)
    decisions = pd.DataFrame(decision_rows)
    summary = _policy_summary(results)
    paired = _paired_comparison_to_reference(
        results,
        "Always NT",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    best_fixed = _best_fixed_hindsight(results, fixed_policy_names)
    adaptive_rows = results[results["policy"] == adaptive_policy_name][
        ["scenario", "intercepted", "escaped"]
    ].rename(
        columns={
            "intercepted": "adaptive_intercepted",
            "escaped": "adaptive_escaped",
        }
    )
    adaptive_vs_best_fixed = adaptive_rows.merge(best_fixed, on="scenario", how="inner")
    adaptive_vs_best_fixed["intercepted_difference"] = (
        adaptive_vs_best_fixed["adaptive_intercepted"]
        - adaptive_vs_best_fixed["best_fixed_intercepted"]
    )
    adaptive_vs_best_fixed["escaped_difference"] = (
        adaptive_vs_best_fixed["adaptive_escaped"]
        - adaptive_vs_best_fixed["best_fixed_escaped"]
    )
    usage = _heuristic_usage(decisions)

    results.to_csv(output_dir / "closed_loop_policy_results_by_scenario.csv", index=False)
    summary.to_csv(output_dir / "closed_loop_policy_summary.csv", index=False)
    paired.to_csv(output_dir / "closed_loop_paired_comparison_vs_always_nt.csv", index=False)
    best_fixed.to_csv(output_dir / "best_fixed_hindsight_by_scenario.csv", index=False)
    adaptive_vs_best_fixed.to_csv(
        output_dir / "adaptive_vs_best_fixed_hindsight.csv",
        index=False,
    )
    decisions.to_csv(output_dir / "adaptive_mu_fc_decision_log.csv", index=False)
    usage.to_csv(output_dir / "adaptive_mu_fc_heuristic_usage.csv", index=False)

    manifest = {
        "method": "closed-loop deployment of mu_FC",
        "training_label_semantics": "fixed-continuation regret from existing dataset",
        "decision_epoch": (
            "reselect after selected target is intercepted or crosses the boundary"
        ),
        "always_nt_role": "evaluation baseline only",
        "global_adaptive_oracle_computed": False,
        "regret_threshold": float(selected_threshold),
        "threshold_mode": args.threshold_mode,
        "arguments": vars(args),
        "model_path": str(model_path),
        "configured_model_path": str(configured_model_path),
        "training_metadata": training_metadata,
    }
    with (output_dir / "closed_loop_experiment_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)

    print("\nPolicy summary")
    print("--------------")
    print(summary.to_string(index=False))
    print("\nPaired comparison versus Always NT")
    print("------------------------------------")
    print(paired.to_string(index=False))
    if not usage.empty:
        print("\nAdaptive heuristic usage")
        print("------------------------")
        print(usage.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
