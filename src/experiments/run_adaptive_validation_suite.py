from __future__ import annotations

"""End-to-end focused evaluation of the fixed-continuation selector.

This script is intentionally additive to the original 5,000-scenario study. It
uses a new decision-epoch dataset to:

1. fit nested 100/250/500-scenario learning-curve models;
2. select the strongest fixed validation baseline;
3. tune a conservative regret threshold on fresh validation scenarios;
4. freeze the selected configuration and evaluate it on fresh test scenarios;
5. compare fixed, one-shot, ungated closed-loop, and conservative closed-loop
   policies using paired full-scenario metrics.

The supervised labels remain fixed-continuation regrets. No adaptive relabeling
or policy iteration is performed here.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.experiments.closed_loop_fc_selector import (
    FixedContinuationRegretSelector,
    canonical_code_name,
    display_heuristic_name,
    infer_dataset_heuristics,
    preprocess_rollout_dataset,
    run_closed_loop_selector,
    run_one_shot_selector,
    train_mu_fc_from_dataframe,
)
from src.experiments.evaluate_regret_selector import load_dataset
from src.experiments.run_closed_loop_fc_selector_experiment import (
    _best_fixed_hindsight,
    _bootstrap_mean_ci,
    _heuristic_usage,
    _paired_comparison_to_reference,
    _policy_summary,
    _selector_with_threshold,
    _write_training_outputs,
)
from src.experiments.run_large_scale_rollout import make_large_scale_scenarios
from src.sim.runner import run_episode


def _parse_number_list(value: str, cast: Any) -> List[Any]:
    result: List[Any] = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            result.append(cast(token))
    if not result:
        raise ValueError(f"Expected at least one comma-separated value, got {value!r}.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the focused learning-curve, validation, threshold-selection, "
            "and held-out closed-loop evaluation suite."
        )
    )
    parser.add_argument("--rollout-input-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--dataset-mode",
        choices=["with_ties", "no_ties", "full"],
        default="with_ties",
    )
    parser.add_argument(
        "--training-sizes",
        type=str,
        default="100,250,500",
        help="Nested numbers of independent dataset scenarios.",
    )
    parser.add_argument(
        "--training-size-selection",
        choices=["largest", "validation"],
        default="largest",
        help=(
            "Use all available training scenarios by default. 'validation' "
            "selects the size with the best ungated closed-loop validation mean."
        ),
    )
    parser.add_argument("--scenario-order-seed", type=int, default=20260807)
    parser.add_argument("--internal-validation-size", type=float, default=0.25)
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
        "--keep-duplicate-initial-states",
        action="store_true",
        help="Disable the default one-initial-row-per-scenario preprocessing.",
    )

    parser.add_argument("--n-validation-scenarios", type=int, default=150)
    parser.add_argument("--validation-seed", type=int, default=20260809)
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="0,0.25,0.5,1,1.5,2,3",
    )
    parser.add_argument(
        "--threshold-selection-rule",
        choices=["lower_ci", "mean"],
        default="lower_ci",
        help=(
            "'lower_ci' maximizes the paired 95%% lower confidence bound against "
            "the validation-selected fixed baseline; 'mean' maximizes mean gain."
        ),
    )

    parser.add_argument("--n-test-scenarios", type=int, default=300)
    parser.add_argument("--test-seed", type=int, default=20260810)
    parser.add_argument(
        "--scenario-mix",
        choices=["baseline", "decision_rich", "heavy_load"],
        default="decision_rich",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=731)
    parser.add_argument(
        "--no-decision-log",
        action="store_true",
        help="Skip per-decision logs in the final test only.",
    )
    return parser.parse_args()


def _scenario_label(prefix: str, seed: int, name: str) -> str:
    return f"{prefix}_{seed}_{name}"


def _fixed_result_row(
    scenario: str,
    params: Any,
    heuristic: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "policy": f"Always {display_heuristic_name(heuristic)}",
        "policy_type": "fixed_heuristic",
        "scenario_regime": params.scenario_regime,
        "spatial_structure": params.spatial_structure,
        "arrival_process": params.arrival_process,
        "deadline_pressure": params.deadline_pressure,
        "spawned": int(result["spawned"]),
        "intercepted": int(result["intercepted"]),
        "escaped": int(result["escaped"]),
        "interception_rate": float(result["intercepted"] / max(1, result["spawned"])),
        "num_decisions": np.nan,
        "num_heuristic_switches": 0,
        "num_baseline_overrides": 0,
        "override_share": 0.0,
        "num_threshold_blocked": 0,
    }


def _adaptive_result_row(
    scenario: str,
    params: Any,
    policy_name: str,
    policy_type: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "policy": policy_name,
        "policy_type": policy_type,
        "scenario_regime": params.scenario_regime,
        "spatial_structure": params.spatial_structure,
        "arrival_process": params.arrival_process,
        "deadline_pressure": params.deadline_pressure,
        "spawned": int(result["spawned"]),
        "intercepted": int(result["intercepted"]),
        "escaped": int(result["escaped"]),
        "interception_rate": float(result["interception_rate"]),
        "num_decisions": int(result["num_decisions"]),
        "num_heuristic_switches": int(result["num_heuristic_switches"]),
        "num_baseline_overrides": int(result.get("num_baseline_overrides", 0)),
        "override_share": float(result.get("override_share", 0.0)),
        "num_threshold_blocked": int(result.get("num_threshold_blocked", 0)),
    }


def _evaluate_fixed_policies(
    scenarios: Mapping[str, Any],
    heuristic_names: Sequence[str],
    *,
    prefix: str,
    seed: int,
    desc: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, scenario_obj in tqdm(scenarios.items(), desc=desc):
        scenario = _scenario_label(prefix, seed, name)
        params = scenario_obj.params
        for heuristic in heuristic_names:
            result = run_episode(params, heuristic_name=heuristic, preempt=False)
            rows.append(_fixed_result_row(scenario, params, heuristic, result))
    return pd.DataFrame(rows)


def _evaluate_closed_loop(
    scenarios: Mapping[str, Any],
    selector: FixedContinuationRegretSelector,
    *,
    policy_name: str,
    policy_type: str,
    prefix: str,
    seed: int,
    collect_decisions: bool,
    desc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    for name, scenario_obj in tqdm(scenarios.items(), desc=desc):
        scenario = _scenario_label(prefix, seed, name)
        params = scenario_obj.params
        result = run_closed_loop_selector(
            params,
            selector,
            collect_decisions=collect_decisions,
        )
        result_rows.append(
            _adaptive_result_row(scenario, params, policy_name, policy_type, result)
        )
        for decision in result["decision_log"]:
            decision_rows.append(
                {
                    "scenario": scenario,
                    "policy": policy_name,
                    "scenario_regime": params.scenario_regime,
                    "spatial_structure": params.spatial_structure,
                    "arrival_process": params.arrival_process,
                    "deadline_pressure": params.deadline_pressure,
                    **decision,
                }
            )
    return pd.DataFrame(result_rows), pd.DataFrame(decision_rows)


def _evaluate_one_shot(
    scenarios: Mapping[str, Any],
    selector: FixedContinuationRegretSelector,
    *,
    policy_name: str,
    prefix: str,
    seed: int,
    desc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    for name, scenario_obj in tqdm(scenarios.items(), desc=desc):
        scenario = _scenario_label(prefix, seed, name)
        params = scenario_obj.params
        result = run_one_shot_selector(params, selector, collect_decisions=True)
        result_rows.append(
            _adaptive_result_row(
                scenario,
                params,
                policy_name,
                "learned_one_shot_mu_FC",
                result,
            )
        )
        for decision in result["decision_log"]:
            decision_rows.append(
                {
                    "scenario": scenario,
                    "policy": policy_name,
                    "scenario_regime": params.scenario_regime,
                    "spatial_structure": params.spatial_structure,
                    "arrival_process": params.arrival_process,
                    "deadline_pressure": params.deadline_pressure,
                    **decision,
                }
            )
    return pd.DataFrame(result_rows), pd.DataFrame(decision_rows)


def _select_best_fixed_baseline(
    fixed_results: pd.DataFrame,
    heuristic_names: Sequence[str],
) -> tuple[str, str, pd.DataFrame]:
    summary = _policy_summary(fixed_results)
    fixed_summary = summary[summary["policy"].str.startswith("Always ")].copy()
    fixed_summary = fixed_summary.sort_values(
        ["mean_intercepted", "mean_escaped", "mean_active_at_horizon", "policy"],
        ascending=[False, True, True, True],
    )
    policy_name = str(fixed_summary.iloc[0]["policy"])
    display_name = policy_name.replace("Always ", "", 1)
    code_by_display = {
        display_heuristic_name(code): str(code) for code in heuristic_names
    }
    if display_name not in code_by_display:
        raise ValueError(f"Could not map fixed baseline {policy_name!r} to a code label.")
    return code_by_display[display_name], policy_name, summary


def _paired_metrics(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    merged = candidate[["scenario", "intercepted", "escaped"]].merge(
        baseline[["scenario", "intercepted", "escaped"]],
        on="scenario",
        suffixes=("_candidate", "_baseline"),
        how="inner",
    )
    difference = (
        merged["intercepted_candidate"].to_numpy(dtype=float)
        - merged["intercepted_baseline"].to_numpy(dtype=float)
    )
    escaped_difference = (
        merged["escaped_candidate"].to_numpy(dtype=float)
        - merged["escaped_baseline"].to_numpy(dtype=float)
    )
    low, high = _bootstrap_mean_ci(
        difference,
        n_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "scenarios": int(len(merged)),
        "mean_difference_vs_baseline": float(np.mean(difference)),
        "median_difference_vs_baseline": float(np.median(difference)),
        "ci95_low": low,
        "ci95_high": high,
        "win_rate_vs_baseline": float(np.mean(difference > 0)),
        "tie_rate_vs_baseline": float(np.mean(difference == 0)),
        "loss_rate_vs_baseline": float(np.mean(difference < 0)),
        "mean_escaped_difference_vs_baseline": float(np.mean(escaped_difference)),
    }


def _state_level_fixed_baseline_summary(
    predictions: pd.DataFrame,
    heuristics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {
            "policy": "Fixed-continuation oracle",
            "mean_future_intercepted": float(
                predictions["best_future_intercepted"].mean()
            ),
            "mean_regret": 0.0,
            "zero_regret_rate": 1.0,
        },
        {
            "policy": "Learned selector",
            "mean_future_intercepted": float(
                predictions["model_future_intercepted"].mean()
            ),
            "mean_regret": float(predictions["model_regret"].mean()),
            "zero_regret_rate": float((predictions["model_regret"] == 0).mean()),
        },
    ]
    for heuristic in heuristics:
        rows.append(
            {
                "policy": f"Always {display_heuristic_name(heuristic)}",
                "mean_future_intercepted": float(
                    predictions[f"{heuristic}_future_intercepted"].mean()
                ),
                "mean_regret": float(predictions[f"{heuristic}_regret"].mean()),
                "zero_regret_rate": float(
                    (predictions[f"{heuristic}_regret"] == 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_future_intercepted", ascending=False
    )


def _save_training_artifacts(artifacts: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_training_outputs(artifacts, directory)
    _state_level_fixed_baseline_summary(
        artifacts.validation_predictions,
        artifacts.selector.candidate_heuristics,
    ).to_csv(directory / "state_level_validation_policy_comparison.csv", index=False)


def _choose_training_size(
    summary: pd.DataFrame,
    *,
    mode: str,
) -> int:
    if mode == "largest":
        return int(summary["training_scenarios"].max())
    ranked = summary.sort_values(
        [
            "validation_mean_intercepted",
            "validation_ci95_low_vs_baseline",
            "validation_loss_rate_vs_baseline",
            "training_scenarios",
        ],
        ascending=[False, False, True, False],
    )
    return int(ranked.iloc[0]["training_scenarios"])


def _choose_threshold(summary: pd.DataFrame, rule: str) -> float:
    if rule == "lower_ci":
        ranked = summary.sort_values(
            [
                "ci95_low",
                "mean_difference_vs_baseline",
                "loss_rate_vs_baseline",
                "mean_override_share",
                "threshold",
            ],
            ascending=[False, False, True, True, False],
        )
    else:
        ranked = summary.sort_values(
            [
                "mean_difference_vs_baseline",
                "ci95_low",
                "loss_rate_vs_baseline",
                "mean_override_share",
                "threshold",
            ],
            ascending=[False, False, True, True, False],
        )
    return float(ranked.iloc[0]["threshold"])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_sizes = sorted(set(_parse_number_list(args.training_sizes, int)))
    thresholds = sorted({max(0.0, value) for value in _parse_number_list(args.threshold_grid, float)})

    raw = load_dataset(Path(args.rollout_input_dir), args.dataset_mode)
    prepared, preprocessing_report = preprocess_rollout_dataset(
        raw,
        deduplicate_initial_states=not args.keep_duplicate_initial_states,
    )
    preprocessing_report.to_csv(output_dir / "dataset_preprocessing.csv", index=False)
    prepared.to_csv(output_dir / "prepared_training_dataset.csv", index=False)

    available_scenarios = np.asarray(sorted(prepared["scenario"].astype(str).unique()))
    if max(training_sizes) > len(available_scenarios):
        raise ValueError(
            f"Largest requested training size {max(training_sizes)} exceeds the "
            f"{len(available_scenarios)} available independent scenarios."
        )
    rng = np.random.default_rng(args.scenario_order_seed)
    scenario_order = available_scenarios.copy()
    rng.shuffle(scenario_order)
    pd.DataFrame(
        {
            "scenario_order_index": np.arange(len(scenario_order)),
            "scenario": scenario_order,
        }
    ).to_csv(output_dir / "nested_training_scenario_order.csv", index=False)

    heuristics = infer_dataset_heuristics(prepared)
    validation_scenarios = make_large_scale_scenarios(
        n_scenarios=args.n_validation_scenarios,
        seed=args.validation_seed,
        scenario_mix=args.scenario_mix,
    )
    validation_params = pd.DataFrame(
        [
            {
                "scenario": _scenario_label("validation", args.validation_seed, name),
                "scenario_mix": args.scenario_mix,
                **asdict(obj.params),
            }
            for name, obj in validation_scenarios.items()
        ]
    )
    validation_params.to_csv(output_dir / "validation_scenario_params.csv", index=False)

    fixed_validation = _evaluate_fixed_policies(
        validation_scenarios,
        heuristics,
        prefix="validation",
        seed=args.validation_seed,
        desc="Validation fixed heuristics",
    )
    baseline_code, baseline_policy, fixed_validation_summary = _select_best_fixed_baseline(
        fixed_validation,
        heuristics,
    )
    fixed_validation.to_csv(output_dir / "validation_fixed_results.csv", index=False)
    fixed_validation_summary.to_csv(
        output_dir / "validation_fixed_policy_summary.csv", index=False
    )
    baseline_validation = fixed_validation[
        fixed_validation["policy"] == baseline_policy
    ].copy()
    print(
        "Validation-selected fixed baseline: "
        f"{baseline_policy} (code={baseline_code})"
    )

    model_by_size: Dict[int, FixedContinuationRegretSelector] = {}
    metadata_by_size: Dict[int, Dict[str, Any]] = {}
    learning_rows: List[Dict[str, Any]] = []
    validation_adaptive_rows: List[pd.DataFrame] = []

    for index, size in enumerate(training_sizes):
        print(f"\n=== Learning-curve model: {size} scenarios ===")
        scenario_subset = scenario_order[:size].tolist()
        model_dir = output_dir / "learning_curve" / f"train_{size}"
        artifacts = train_mu_fc_from_dataframe(
            prepared,
            dataset_mode=args.dataset_mode,
            validation_size=args.internal_validation_size,
            random_state=args.random_state,
            sample_weight_mode=args.sample_weight_mode,
            weight_alpha=args.weight_alpha,
            clip_abs=args.clip_abs,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=args.max_depth,
            deduplicate_initial_states=False,
            scenario_subset=scenario_subset,
            source_description=str(args.rollout_input_dir),
        )
        _save_training_artifacts(artifacts, model_dir)
        ungated = _selector_with_threshold(
            artifacts.selector,
            threshold=0.0,
            threshold_mode="none",
            baseline_heuristic=baseline_code,
            name=f"Closed-loop mu_FC ungated (train={size})",
        )
        model_path = model_dir / "mu_fc_ungated.joblib"
        ungated.save(model_path, artifacts.metadata)
        model_by_size[size] = ungated
        metadata_by_size[size] = artifacts.metadata

        adaptive_validation, _ = _evaluate_closed_loop(
            validation_scenarios,
            ungated,
            policy_name=ungated.name,
            policy_type="learning_curve_closed_loop_ungated",
            prefix="validation",
            seed=args.validation_seed,
            collect_decisions=False,
            desc=f"Validation closed-loop train={size}",
        )
        validation_adaptive_rows.append(adaptive_validation)
        paired = _paired_metrics(
            adaptive_validation,
            baseline_validation,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + index,
        )
        internal = artifacts.validation_summary.iloc[0].to_dict()
        learning_rows.append(
            {
                "training_scenarios": int(size),
                "training_rows": int(artifacts.metadata["rows"]),
                "internal_validation_rows": int(internal["test_rows"]),
                "internal_mean_model_regret": float(internal["mean_model_regret"]),
                "internal_zero_regret_rate": float(internal["zero_regret_rate"]),
                "validation_mean_intercepted": float(
                    adaptive_validation["intercepted"].mean()
                ),
                "validation_mean_escaped": float(
                    adaptive_validation["escaped"].mean()
                ),
                "validation_mean_switches": float(
                    adaptive_validation["num_heuristic_switches"].mean()
                ),
                "validation_ci95_low_vs_baseline": paired["ci95_low"],
                "validation_ci95_high_vs_baseline": paired["ci95_high"],
                "validation_mean_difference_vs_baseline": paired[
                    "mean_difference_vs_baseline"
                ],
                "validation_win_rate_vs_baseline": paired[
                    "win_rate_vs_baseline"
                ],
                "validation_tie_rate_vs_baseline": paired[
                    "tie_rate_vs_baseline"
                ],
                "validation_loss_rate_vs_baseline": paired[
                    "loss_rate_vs_baseline"
                ],
                "validation_fixed_baseline": baseline_policy,
                "model_path": str(model_path),
            }
        )

    learning_curve = pd.DataFrame(learning_rows)
    learning_curve.to_csv(output_dir / "learning_curve_summary.csv", index=False)
    pd.concat(validation_adaptive_rows, ignore_index=True).to_csv(
        output_dir / "learning_curve_validation_results.csv", index=False
    )
    selected_size = _choose_training_size(
        learning_curve,
        mode=args.training_size_selection,
    )
    selected_ungated = model_by_size[selected_size]
    print(f"Selected training size: {selected_size}")

    threshold_rows: List[Dict[str, Any]] = []
    threshold_validation_results: List[pd.DataFrame] = []
    for offset, threshold in enumerate(thresholds):
        candidate = _selector_with_threshold(
            selected_ungated,
            threshold=threshold,
            threshold_mode="baseline_override",
            baseline_heuristic=baseline_code,
            name=(
                f"Conservative closed-loop mu_FC "
                f"(base={display_heuristic_name(baseline_code)}, tau={threshold:g})"
            ),
        )
        candidate_results, _ = _evaluate_closed_loop(
            validation_scenarios,
            candidate,
            policy_name=candidate.name,
            policy_type="threshold_validation_closed_loop",
            prefix="validation",
            seed=args.validation_seed,
            collect_decisions=False,
            desc=f"Threshold validation tau={threshold:g}",
        )
        threshold_validation_results.append(candidate_results)
        paired = _paired_metrics(
            candidate_results,
            baseline_validation,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + 100 + offset,
        )
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "baseline_code": baseline_code,
                "baseline_policy": baseline_policy,
                "mean_intercepted": float(candidate_results["intercepted"].mean()),
                "mean_escaped": float(candidate_results["escaped"].mean()),
                "mean_active_at_horizon": float(
                    (
                        candidate_results["spawned"]
                        - candidate_results["intercepted"]
                        - candidate_results["escaped"]
                    ).mean()
                ),
                "mean_switches": float(
                    candidate_results["num_heuristic_switches"].mean()
                ),
                "mean_override_share": float(
                    candidate_results["override_share"].mean()
                ),
                **paired,
            }
        )

    threshold_summary = pd.DataFrame(threshold_rows)
    threshold_summary.to_csv(output_dir / "threshold_validation_summary.csv", index=False)
    pd.concat(threshold_validation_results, ignore_index=True).to_csv(
        output_dir / "threshold_validation_results_by_scenario.csv", index=False
    )
    selected_threshold = _choose_threshold(
        threshold_summary,
        args.threshold_selection_rule,
    )
    print(f"Selected regret threshold: {selected_threshold:g}")

    configured = _selector_with_threshold(
        selected_ungated,
        threshold=selected_threshold,
        threshold_mode="baseline_override",
        baseline_heuristic=baseline_code,
        name=(
            f"Conservative closed-loop mu_FC "
            f"(base={display_heuristic_name(baseline_code)}, tau={selected_threshold:g})"
        ),
    )
    selected_ungated_path = output_dir / "selected_mu_fc_ungated.joblib"
    configured_path = output_dir / "selected_mu_fc_configured.joblib"
    selected_ungated.save(
        selected_ungated_path,
        {
            **metadata_by_size[selected_size],
            "selected_training_size": selected_size,
            "validation_fixed_baseline": baseline_code,
        },
    )
    configured.save(
        configured_path,
        {
            **metadata_by_size[selected_size],
            "selected_training_size": selected_size,
            "validation_fixed_baseline": baseline_code,
            "selected_threshold": selected_threshold,
            "threshold_selection_rule": args.threshold_selection_rule,
        },
    )

    test_scenarios = make_large_scale_scenarios(
        n_scenarios=args.n_test_scenarios,
        seed=args.test_seed,
        scenario_mix=args.scenario_mix,
    )
    pd.DataFrame(
        [
            {
                "scenario": _scenario_label("test", args.test_seed, name),
                "scenario_mix": args.scenario_mix,
                **asdict(obj.params),
            }
            for name, obj in test_scenarios.items()
        ]
    ).to_csv(output_dir / "test_scenario_params.csv", index=False)

    test_fixed = _evaluate_fixed_policies(
        test_scenarios,
        heuristics,
        prefix="test",
        seed=args.test_seed,
        desc="Test fixed heuristics",
    )
    one_shot_name = "One-shot mu_FC (ungated)"
    test_one_shot, one_shot_log = _evaluate_one_shot(
        test_scenarios,
        selected_ungated,
        policy_name=one_shot_name,
        prefix="test",
        seed=args.test_seed,
        desc="Test one-shot mu_FC",
    )
    ungated_name = "Closed-loop mu_FC (ungated)"
    test_ungated, ungated_log = _evaluate_closed_loop(
        test_scenarios,
        selected_ungated,
        policy_name=ungated_name,
        policy_type="adaptive_closed_loop_mu_FC_ungated",
        prefix="test",
        seed=args.test_seed,
        collect_decisions=not args.no_decision_log,
        desc="Test closed-loop mu_FC ungated",
    )
    conservative_name = configured.name
    test_conservative, conservative_log = _evaluate_closed_loop(
        test_scenarios,
        configured,
        policy_name=conservative_name,
        policy_type="adaptive_closed_loop_mu_FC_conservative",
        prefix="test",
        seed=args.test_seed,
        collect_decisions=not args.no_decision_log,
        desc="Test conservative closed-loop mu_FC",
    )

    test_results = pd.concat(
        [test_fixed, test_one_shot, test_ungated, test_conservative],
        ignore_index=True,
    )
    test_summary = _policy_summary(test_results)
    paired_vs_baseline = _paired_comparison_to_reference(
        test_results,
        baseline_policy,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    paired_vs_nt = _paired_comparison_to_reference(
        test_results,
        "Always NT",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 1000,
    )
    fixed_policy_names = [
        f"Always {display_heuristic_name(code)}" for code in heuristics
    ]
    best_fixed = _best_fixed_hindsight(test_results, fixed_policy_names)

    adaptive_comparisons: List[pd.DataFrame] = []
    for policy_name in [one_shot_name, ungated_name, conservative_name]:
        adaptive_rows = test_results[test_results["policy"] == policy_name][
            ["scenario", "intercepted", "escaped"]
        ].rename(
            columns={
                "intercepted": "adaptive_intercepted",
                "escaped": "adaptive_escaped",
            }
        )
        comparison = adaptive_rows.merge(best_fixed, on="scenario", how="inner")
        comparison.insert(1, "policy", policy_name)
        comparison["intercepted_difference"] = (
            comparison["adaptive_intercepted"]
            - comparison["best_fixed_intercepted"]
        )
        comparison["escaped_difference"] = (
            comparison["adaptive_escaped"] - comparison["best_fixed_escaped"]
        )
        adaptive_comparisons.append(comparison)

    decision_logs = pd.concat(
        [one_shot_log, ungated_log, conservative_log],
        ignore_index=True,
        sort=False,
    )
    usage = _heuristic_usage(decision_logs)
    if not decision_logs.empty:
        usage = (
            decision_logs.groupby(["policy", "heuristic"])
            .agg(
                decisions=("heuristic", "size"),
                scenarios_used=("scenario", "nunique"),
            )
            .reset_index()
        )
        usage["decision_share_within_policy"] = usage.groupby("policy")[
            "decisions"
        ].transform(lambda values: values / max(1, values.sum()))

    test_results.to_csv(output_dir / "test_policy_results_by_scenario.csv", index=False)
    test_summary.to_csv(output_dir / "test_policy_summary.csv", index=False)
    paired_vs_baseline.to_csv(
        output_dir / "test_paired_comparison_vs_validation_baseline.csv", index=False
    )
    paired_vs_nt.to_csv(output_dir / "test_paired_comparison_vs_nt.csv", index=False)
    best_fixed.to_csv(output_dir / "test_best_fixed_hindsight.csv", index=False)
    pd.concat(adaptive_comparisons, ignore_index=True).to_csv(
        output_dir / "test_adaptive_vs_best_fixed_hindsight.csv", index=False
    )
    decision_logs.to_csv(output_dir / "test_adaptive_decision_log.csv", index=False)
    usage.to_csv(output_dir / "test_adaptive_heuristic_usage.csv", index=False)

    manifest = {
        "method": "focused additive adaptive validation suite",
        "training_label_semantics": "fixed-continuation regret",
        "dataset_mode": args.dataset_mode,
        "initial_state_deduplication": not args.keep_duplicate_initial_states,
        "training_sizes": training_sizes,
        "selected_training_size": selected_size,
        "training_size_selection": args.training_size_selection,
        "validation_fixed_baseline_code": baseline_code,
        "validation_fixed_baseline_policy": baseline_policy,
        "threshold_grid": thresholds,
        "selected_threshold": selected_threshold,
        "threshold_selection_rule": args.threshold_selection_rule,
        "test_policies": [
            *fixed_policy_names,
            one_shot_name,
            ungated_name,
            conservative_name,
        ],
        "exact_global_oracle_in_this_script": False,
        "selected_model": str(configured_path),
        "arguments": vars(args),
    }
    with (output_dir / "adaptive_validation_suite_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)

    print("\nLearning curve")
    print("--------------")
    print(learning_curve.to_string(index=False))
    print("\nThreshold validation")
    print("--------------------")
    print(threshold_summary.to_string(index=False))
    print("\nHeld-out test policy summary")
    print("----------------------------")
    print(test_summary.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
