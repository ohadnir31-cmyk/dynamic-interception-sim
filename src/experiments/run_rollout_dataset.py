from __future__ import annotations

import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.sim.env import ScenarioParams
from src.experiments.canonical_scenarios import get_canonical_scenarios
from src.experiments.rollout_labeling import generate_dataset_for_scenarios


def make_complex_stochastic_scenarios() -> dict[str, Any]:
    """
    More complex scenario generator.

    Goal:
    - more active targets
    - harder decisions
    - fewer trivial ties
    - clearer differences between heuristics
    """

    scenarios = {}

    # seeds = range(6)
    seeds = range(3)

    # Higher arrival rates -> more simultaneous targets
    # lambdas = [0.7, 0.9, 1.1]
    lambdas = [0.7, 1.0]

    # Targets spawn farther from boundary, giving time for interactions
    # x_means = [35.0, 50.0, 65.0]
    x_means = [40.0, 60.0]

    # Wider spatial spread
    # y_sigmas = [25.0, 40.0, 60.0]
    y_sigmas = [30.0, 50.0]

    # More speed diversity
    # v_threats = [10.0, 14.0, 18.0]
    # v_stds = [2.0, 4.0]
    v_threats = [12.0, 17.0]
    v_stds = [3.0]

    # Interceptor sometimes slower / comparable / faster
    # v_interceptors = [14.0, 18.0, 22.0]
    v_interceptors = [16.0, 22.0]

    idx = 0

    for seed in seeds:
        for lam in lambdas:
            for x_mean in x_means:
                for y_sigma in y_sigmas:
                    for v_threat in v_threats:
                        for v_std in v_stds:
                            for v_interceptor in v_interceptors:
                                name = f"complex_{idx:05d}"

                                scenarios[name] = type(
                                    "ScenarioObj",
                                    (),
                                    {
                                        "params": ScenarioParams(
                                            seed=seed,
                                            # horizon_T=75.0,
                                            horizon_T=55.0,
                                            dt=0.5,
                                            lambda_arrival=lam,
                                            x_spawn_mean=x_mean,
                                            x_spawn_std=12.0,
                                            y_spawn_sigma=y_sigma,
                                            v_threat_mean=v_threat,
                                            v_threat_std=v_std,
                                            v_interceptor=v_interceptor,
                                            kill_radius=2.0,
                                            home=(0.0, 0.0),
                                            manual_threats=None,
                                        )
                                    },
                                )()

                                idx += 1

    return scenarios


def filter_informative_states(
    df_rollout: pd.DataFrame,
    candidate_heuristics: list[str],
    keep_ties: bool = True,
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


def build_rollout_dataset_complex(
    include_canonical: bool = True,
    include_complex_stochastic: bool = True,
    max_states_per_run: int | None = 20,
    output_prefix: str = "rollout_dataset_compact_heuristics_complex",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_start = time.time()

    # Compact, qualitatively distinct heuristic set
    candidate_heuristics = [
        "NI",
        "MPS",
        "FNI",
        "Ratio",
        "Danger",
        "Cluster",
    ]

    # Behavior policies used to generate state distribution
    behavior_heuristics = ["NI", "Ratio", "Cluster"]
    # behavior_heuristics = [
    #     "NI",
    #     "MPS",
    #     "FNI",
    #     "Ratio",
    #     "Danger",
    #     "Cluster",
    # ]

    scenarios: dict[str, Any] = {}

    if include_canonical:
        scenarios.update(get_canonical_scenarios())

    if include_complex_stochastic:
        scenarios.update(make_complex_stochastic_scenarios())

    scenario_items = list(scenarios.items())
    total_scenarios = len(scenario_items)

    print("\n=== Rollout Dataset Build: Compact Heuristics + Complex Scenarios ===")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Canonical scenarios: {include_canonical}")
    print(f"Complex stochastic scenarios: {include_complex_stochastic}")
    print(f"Candidate heuristics: {candidate_heuristics}")
    print(f"Behavior heuristics: {behavior_heuristics}")
    print(f"Max states per run: {max_states_per_run}")
    print(f"Output prefix: {output_prefix}")
    print()

    all_parts: list[pd.DataFrame] = []
    loop_start = time.time()

    for i, (scenario_name, scenario_obj) in enumerate(
        tqdm(scenario_items, desc="Processing scenarios"),
        start=1,
    ):
        df_part = generate_dataset_for_scenarios(
            scenarios={scenario_name: scenario_obj},
            behavior_heuristics=behavior_heuristics,
            candidate_heuristics=candidate_heuristics,
            rollout_preempt=False,
            max_states_per_run=max_states_per_run,
        )

        all_parts.append(df_part)

        elapsed = time.time() - loop_start
        avg_per_scenario = elapsed / i
        remaining = avg_per_scenario * (total_scenarios - i)

        if i == 1 or i % 25 == 0 or i == total_scenarios:
            rows_so_far = sum(len(part) for part in all_parts)

            print(
                f"\n[{i}/{total_scenarios}] "
                f"Rows so far: {rows_so_far} | "
                f"Elapsed: {elapsed / 60:.2f} min | "
                f"ETA: {remaining / 60:.2f} min | "
                f"Avg/scenario: {avg_per_scenario:.2f} sec"
            )

            # checkpoint
            temp_df = pd.concat(all_parts, ignore_index=True)
            temp_df.to_csv(f"{output_prefix}_checkpoint.csv", index=False)
            print(f"Checkpoint saved: {output_prefix}_checkpoint.csv")

    if all_parts:
        df_rollout = pd.concat(all_parts, ignore_index=True)
    else:
        df_rollout = pd.DataFrame()

    df_informative_with_ties = filter_informative_states(
        df_rollout=df_rollout,
        candidate_heuristics=candidate_heuristics,
        keep_ties=True,
    )

    df_informative_no_ties = filter_informative_states(
        df_rollout=df_rollout,
        candidate_heuristics=candidate_heuristics,
        keep_ties=False,
    )

    full_path = f"{output_prefix}.csv"
    informative_ties_path = f"{output_prefix}_informative_with_ties.csv"
    informative_no_ties_path = f"{output_prefix}_informative_no_ties.csv"

    df_rollout.to_csv(full_path, index=False)
    df_informative_with_ties.to_csv(informative_ties_path, index=False)
    df_informative_no_ties.to_csv(informative_no_ties_path, index=False)

    total_elapsed = time.time() - global_start

    print("\n=== FINAL SUMMARY ===")
    print(f"Total runtime: {total_elapsed:.2f} sec")
    print(f"Total runtime: {total_elapsed / 60:.2f} min")

    print("\n=== Full rollout dataset ===")
    print(df_rollout.shape)
    if not df_rollout.empty:
        print(df_rollout["winner"].value_counts())

    print("\n=== Winner sets ===")
    if not df_rollout.empty:
        print(df_rollout["winner_set"].value_counts())

    print("\n=== Informative states, with ties ===")
    print(df_informative_with_ties.shape)
    if not df_informative_with_ties.empty:
        print(df_informative_with_ties["winner"].value_counts())
        print(df_informative_with_ties["winner_set"].value_counts())

    print("\n=== Informative states, no ties ===")
    print(df_informative_no_ties.shape)
    if not df_informative_no_ties.empty:
        print(df_informative_no_ties["winner"].value_counts())
        print(df_informative_no_ties["winner_set"].value_counts())

    print("\n=== Saved files ===")
    print(f"- {full_path}")
    print(f"- {informative_ties_path}")
    print(f"- {informative_no_ties_path}")
    print(f"- {output_prefix}_checkpoint.csv")

    return df_rollout, df_informative_with_ties, df_informative_no_ties


def main() -> None:
    build_rollout_dataset_complex(
        include_canonical=True,
        include_complex_stochastic=True,
        max_states_per_run=10,
        # max_states_per_run=20,
        output_prefix="rollout_dataset_compact_heuristics_complex",
    )


if __name__ == "__main__":
    main()
