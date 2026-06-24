from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.experiments.rollout_labeling import (
    generate_dataset_for_scenarios,
    n_active_bucket,
)
from src.sim.env import ScenarioParams
from src.sim.runner import run_episode


class ScenarioObj:
    def __init__(self, params: ScenarioParams):
        self.params = params


CANDIDATE_HEURISTICS = ["NI", "FNI", "FMTTB", "MPS", "FCluster"]

DEFAULT_BEHAVIOR_HEURISTICS = ["NI", "FNI", "FMTTB", "MPS", "FCluster"]


# Scenario-mix weights control how often each load regime is sampled.
#
# baseline:
#     The original broad distribution used in the lead-pursuit experiments.
# decision_rich:
#     Increases the prevalence of medium/high-load regimes where the choice
#     between heuristics is expected to be more informative, without letting
#     overloaded cases dominate the entire dataset.
# heavy_load:
#     A stronger stress-test distribution for later sensitivity experiments.
SCENARIO_MIX_WEIGHTS: Dict[str, Dict[str, float]] = {
    "baseline": {
        "low_load": 0.15,
        "medium_load": 0.25,
        "high_load": 0.35,
        "overloaded": 0.25,
    },
    "decision_rich": {
        "low_load": 0.05,
        "medium_load": 0.25,
        "high_load": 0.40,
        "overloaded": 0.30,
    },
    "heavy_load": {
        "low_load": 0.02,
        "medium_load": 0.13,
        "high_load": 0.35,
        "overloaded": 0.50,
    },
}


REGIME_CONFIG: Dict[str, Dict[str, Any]] = {
    "low_load": {
        "weight": 0.15,
        "initial_targets": (2, 4),
        "lambda_arrival": (0.08, 0.15),
        "x_spawn_mean": (8.0, 13.0),
        "x_spawn_std": (1.5, 3.0),
        "y_spawn_sigma": (3.0, 7.0),
        "v_threat_mean": (0.15, 0.30),
        "v_threat_std": (0.03, 0.07),
    },
    "medium_load": {
        "weight": 0.25,
        "initial_targets": (4, 8),
        "lambda_arrival": (0.20, 0.40),
        "x_spawn_mean": (7.0, 12.0),
        "x_spawn_std": (1.5, 3.0),
        "y_spawn_sigma": (4.0, 9.0),
        "v_threat_mean": (0.20, 0.45),
        "v_threat_std": (0.04, 0.10),
    },
    "high_load": {
        "weight": 0.35,
        "initial_targets": (8, 14),
        "lambda_arrival": (0.45, 0.80),
        "x_spawn_mean": (6.0, 11.0),
        "x_spawn_std": (1.5, 3.5),
        "y_spawn_sigma": (5.0, 11.0),
        "v_threat_mean": (0.25, 0.60),
        "v_threat_std": (0.05, 0.12),
    },
    "overloaded": {
        "weight": 0.25,
        "initial_targets": (14, 24),
        "lambda_arrival": (0.80, 1.40),
        "x_spawn_mean": (5.0, 10.0),
        "x_spawn_std": (1.5, 3.5),
        "y_spawn_sigma": (6.0, 12.0),
        "v_threat_mean": (0.30, 0.75),
        "v_threat_std": (0.06, 0.15),
    },
}


def _sample_float(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _sample_int(rng: np.random.Generator, lo_hi: tuple[int, int]) -> int:
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _weighted_choice(rng: np.random.Generator, weights_by_name: Dict[str, float]) -> str:
    names = list(weights_by_name.keys())
    weights = np.array([weights_by_name[name] for name in names], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(names, p=weights))


def _get_scenario_mix_weights(scenario_mix: str) -> Dict[str, float]:
    if scenario_mix not in SCENARIO_MIX_WEIGHTS:
        valid = ", ".join(sorted(SCENARIO_MIX_WEIGHTS))
        raise ValueError(f"Unknown scenario_mix={scenario_mix!r}. Valid values: {valid}")

    weights = dict(SCENARIO_MIX_WEIGHTS[scenario_mix])
    missing = set(REGIME_CONFIG) - set(weights)
    extra = set(weights) - set(REGIME_CONFIG)

    if missing or extra:
        raise ValueError(
            "Scenario mix weights must match REGIME_CONFIG keys. "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError(f"Scenario mix {scenario_mix!r} has non-positive total weight")

    return {name: float(w) / total for name, w in weights.items()}


def make_large_scale_scenarios(
    n_scenarios: int,
    seed: int = 42,
    scenario_mix: str = "baseline",
) -> Dict[str, ScenarioObj]:
    """
    Build a scenario set with enough active targets to make heuristic choice meaningful.

    Normalized formulation:
    interceptor speed = 1.0
    target speeds are sampled relative to the interceptor speed.
    """

    rng = np.random.default_rng(seed)
    regime_weights = _get_scenario_mix_weights(scenario_mix)

    scenarios: Dict[str, ScenarioObj] = {}

    for scenario_id in range(n_scenarios):
        regime = _weighted_choice(rng, regime_weights)
        cfg = REGIME_CONFIG[regime]

        spatial_structure = "clustered" if rng.random() < 0.50 else "uniform"
        arrival_process = "bursty" if rng.random() < 0.40 else "poisson"
        deadline_pressure = "tight" if rng.random() < 0.50 else "moderate"

        x_mean = _sample_float(rng, cfg["x_spawn_mean"])
        v_mean = _sample_float(rng, cfg["v_threat_mean"])

        # Tight deadlines: targets are closer to the boundary and slightly faster.
        if deadline_pressure == "tight":
            x_mean *= float(rng.uniform(0.75, 0.90))
            v_mean *= float(rng.uniform(1.05, 1.25))

        params = ScenarioParams(
            seed=seed * 1_000_000 + scenario_id,
            horizon_T=60.0,
            dt=0.25,
            lambda_arrival=_sample_float(rng, cfg["lambda_arrival"]),
            x_spawn_mean=x_mean,
            x_spawn_std=_sample_float(rng, cfg["x_spawn_std"]),
            y_spawn_sigma=_sample_float(rng, cfg["y_spawn_sigma"]),
            v_threat_mean=v_mean,
            v_threat_std=_sample_float(rng, cfg["v_threat_std"]),
            initial_targets=_sample_int(rng, cfg["initial_targets"]),
            arrival_process=arrival_process,
            spatial_structure=spatial_structure,
            n_clusters=_sample_int(rng, (2, 5)),
            cluster_std=_sample_float(rng, (0.5, 1.8)),
            burst_probability=(
                _sample_float(rng, (0.03, 0.10))
                if arrival_process == "bursty"
                else 0.0
            ),
            burst_size_min=3,
            burst_size_max=_sample_int(rng, (5, 10)),
            scenario_regime=regime,
            deadline_pressure=deadline_pressure,
            v_interceptor=1.0,
            kill_radius=0.12,
            home=(0.0, 0.0),
            manual_threats=None,
        )

        scenario_name = (
            f"large_{scenario_id:05d}_{regime}_{spatial_structure}_"
            f"{arrival_process}_{deadline_pressure}"
        )

        scenarios[scenario_name] = ScenarioObj(params=params)

    return scenarios


def run_full_heuristic_rollouts(
    scenarios: Dict[str, ScenarioObj],
    heuristics: Sequence[str],
    output_path: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for scenario_name, scenario_obj in tqdm(
        scenarios.items(),
        desc="Full heuristic rollouts",
    ):
        params = scenario_obj.params

        for heuristic in heuristics:
            result = run_episode(
                params=params,
                heuristic_name=heuristic,
                preempt=False,
            )

            rows.append(
                {
                    "scenario": scenario_name,
                    "scenario_regime": params.scenario_regime,
                    "spatial_structure": params.spatial_structure,
                    "arrival_process": params.arrival_process,
                    "deadline_pressure": params.deadline_pressure,
                    "initial_targets": params.initial_targets,
                    "lambda_arrival": params.lambda_arrival,
                    "heuristic": heuristic,
                    "spawned": result["spawned"],
                    "intercepted": result["intercepted"],
                    "penetrated": result["escaped"],
                    "interception_rate": result["intercepted"] / max(1, result["spawned"]),
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def summarize_heuristics(
    df_summary: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    rows = []

    for heuristic, g in df_summary.groupby("heuristic"):
        rows.append(
            {
                "heuristic": heuristic,
                "mean_intercepted": g["intercepted"].mean(),
                "std_intercepted": g["intercepted"].std(),
                "mean_penetrated": g["penetrated"].mean(),
                "mean_interception_rate": g["interception_rate"].mean(),
                "num_rollouts": len(g),
            }
        )

    out = pd.DataFrame(rows).sort_values("mean_intercepted", ascending=False)
    out.to_csv(output_path, index=False)
    return out


def summarize_scenario_winners(
    df_summary: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    rows = []

    for scenario, g in df_summary.groupby("scenario"):
        best_intercepted = g["intercepted"].max()
        candidates = g[g["intercepted"] == best_intercepted]

        best_penetrated = candidates["penetrated"].min()
        winners = sorted(
            candidates[candidates["penetrated"] == best_penetrated]["heuristic"].tolist()
        )

        first = g.iloc[0]

        rows.append(
            {
                "scenario": scenario,
                "scenario_regime": first["scenario_regime"],
                "spatial_structure": first["spatial_structure"],
                "arrival_process": first["arrival_process"],
                "deadline_pressure": first["deadline_pressure"],
                "winner": winners[0] if len(winners) == 1 else "TIE",
                "winner_set": ",".join(winners),
                "n_winners": len(winners),
                "best_intercepted": best_intercepted,
                "best_penetrated": best_penetrated,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def filter_informative_states(
    df_rollout: pd.DataFrame,
    candidate_heuristics: Sequence[str],
    keep_ties: bool,
) -> pd.DataFrame:
    if df_rollout.empty:
        return df_rollout.copy()

    df = df_rollout[
        (df_rollout["N_active"] >= 2)
        & (df_rollout["n_winners"] < len(candidate_heuristics))
    ].copy()

    if not keep_ties:
        df = df[df["winner"] != "TIE"].copy()

    return df


def summarize_by_active_targets(
    df_states: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if df_states.empty:
        out = pd.DataFrame()
        out.to_csv(output_path, index=False)
        return out

    df = df_states.copy()

    if "N_active_bucket" not in df.columns:
        df["N_active_bucket"] = df["N_active"].apply(lambda x: n_active_bucket(int(x)))

    rows = []
    bucket_order = ["1", "2-3", "4-6", "7-10", "11+"]

    for bucket in bucket_order:
        g = df[df["N_active_bucket"] == bucket]
        if g.empty:
            continue

        winner_counts = g["winner"].value_counts()
        dominant = winner_counts.index[0]

        rows.append(
            {
                "N_active_bucket": bucket,
                "num_states": len(g),
                "mean_N_active": g["N_active"].mean(),
                "tie_rate": float((g["winner"] == "TIE").mean()),
                "mean_n_winners": g["n_winners"].mean(),
                "dominant_winner": dominant,
                "dominant_winner_share": float(winner_counts.iloc[0] / len(g)),
                "mean_best_future_intercepted": g["best_future_intercepted"].mean(),
                "mean_best_future_escaped": g["best_future_escaped"].mean(),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the large-scale rollout experiment."
    )

    parser.add_argument("--n-scenarios", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario-mix",
        type=str,
        default="baseline",
        choices=sorted(SCENARIO_MIX_WEIGHTS.keys()),
        help=(
            "Scenario mixture for load-regime sampling. "
            "Use 'decision_rich' to emphasize medium/high-load states where "
            "adaptive heuristic selection is more informative."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="outputs/large_scale")
    parser.add_argument("--max-states-per-run", type=int, default=8)

    parser.add_argument(
        "--state-label-scenarios",
        type=int,
        default=20_000,
        help="Number of scenarios used for state-level counterfactual labels.",
    )

    parser.add_argument(
        "--skip-state-labels",
        action="store_true",
        help="Only run fixed-heuristic scenario rollouts and skip state-level labels.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    print("\n=== Large-Scale Rollout Experiment ===")
    print(f"n_scenarios: {args.n_scenarios}")
    print(f"scenario_mix: {args.scenario_mix}")
    print(f"scenario_mix_weights: {_get_scenario_mix_weights(args.scenario_mix)}")
    print(f"candidate heuristics: {CANDIDATE_HEURISTICS}")
    print(
        "expected full heuristic rollouts: "
        f"{args.n_scenarios * len(CANDIDATE_HEURISTICS):,}"
    )
    print(f"output_dir: {output_dir}")

    scenarios = make_large_scale_scenarios(
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        scenario_mix=args.scenario_mix,
    )

    scenario_params = pd.DataFrame(
        [
            {
                "scenario": name,
                "scenario_mix": args.scenario_mix,
                **asdict(obj.params),
            }
            for name, obj in scenarios.items()
        ]
    )
    scenario_params.to_csv(
        output_dir / "large_scale_scenario_params.csv",
        index=False,
    )

    df_full = run_full_heuristic_rollouts(
        scenarios=scenarios,
        heuristics=CANDIDATE_HEURISTICS,
        output_path=output_dir / "large_scale_full_heuristic_rollouts.csv",
    )

    summarize_heuristics(
        df_full,
        output_dir / "large_scale_heuristic_summary.csv",
    )

    summarize_scenario_winners(
        df_full,
        output_dir / "large_scale_scenario_winners.csv",
    )

    if not args.skip_state_labels:
        label_items = list(scenarios.items())[: args.state_label_scenarios]
        label_scenarios = dict(label_items)

        print("\n=== State-level counterfactual rollout labels ===")
        print(f"state_label_scenarios: {len(label_scenarios)}")
        print(f"behavior heuristics: {DEFAULT_BEHAVIOR_HEURISTICS}")
        print(f"max_states_per_run: {args.max_states_per_run}")

        df_states = generate_dataset_for_scenarios(
            scenarios=label_scenarios,
            behavior_heuristics=DEFAULT_BEHAVIOR_HEURISTICS,
            candidate_heuristics=CANDIDATE_HEURISTICS,
            rollout_preempt=False,
            max_states_per_run=args.max_states_per_run,
        )

        df_states.to_csv(
            output_dir / "large_scale_rollout_states.csv",
            index=False,
        )

        df_with_ties = filter_informative_states(
            df_states,
            CANDIDATE_HEURISTICS,
            keep_ties=True,
        )

        df_no_ties = filter_informative_states(
            df_states,
            CANDIDATE_HEURISTICS,
            keep_ties=False,
        )

        df_with_ties.to_csv(
            output_dir / "large_scale_rollout_states_informative_with_ties.csv",
            index=False,
        )

        df_no_ties.to_csv(
            output_dir / "large_scale_rollout_states_informative_no_ties.csv",
            index=False,
        )

        summarize_by_active_targets(
            df_states=df_states,
            output_path=output_dir / "large_scale_active_targets_analysis.csv",
        )

    elapsed = time.time() - start

    print("\n=== DONE ===")
    print(f"Total runtime: {elapsed / 60:.2f} min")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
