from __future__ import annotations

import pandas as pd

from src.sim.env import ScenarioParams
from src.experiments.canonical_scenarios import get_canonical_scenarios
from src.experiments.rollout_labeling import generate_dataset_for_scenarios


def make_stochastic_scenarios() -> dict:
    scenarios = {}

    seeds = range(20)  # בהמשך אפשר 100+
    lambdas = [0.25, 0.4, 0.6]
    x_means = [25.0, 35.0, 45.0]
    y_sigmas = [15.0, 25.0, 35.0]
    v_threats = [10.0, 13.0, 16.0]
    v_interceptors = [14.0, 16.0, 18.0]

    idx = 0
    for seed in seeds:
        for lam in lambdas:
            for x_mean in x_means:
                for y_sigma in y_sigmas:
                    for v_threat in v_threats:
                        for v_interceptor in v_interceptors:
                            name = f"stoch_{idx:05d}"
                            scenarios[name] = type("ScenarioObj", (), {
                                "params": ScenarioParams(
                                    seed=seed,
                                    horizon_T=60.0,
                                    dt=0.5,
                                    lambda_arrival=lam,
                                    x_spawn_mean=x_mean,
                                    x_spawn_std=8.0,
                                    y_spawn_sigma=y_sigma,
                                    v_threat_mean=v_threat,
                                    v_threat_std=2.0,
                                    v_interceptor=v_interceptor,
                                    kill_radius=2.0,
                                    home=(0.0, 0.0),
                                    manual_threats=None,
                                )
                            })()
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


def build_rollout_dataset(
    include_canonical: bool = True,
    include_stochastic: bool = True,
    max_states_per_run: int | None = 50,
    output_prefix: str = "rollout_labeled_dataset_v2",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_heuristics = ["NI", "TTB", "MPS", "Weighted", "Cluster"]
    behavior_heuristics = ["NI", "MPS", "Cluster"]

    scenarios = {}

    if include_canonical:
        scenarios.update(get_canonical_scenarios())

    if include_stochastic:
        scenarios.update(make_stochastic_scenarios())

    print(f"Total scenarios: {len(scenarios)}")
    print(f"Behavior heuristics: {behavior_heuristics}")
    print(f"Candidate heuristics: {candidate_heuristics}")
    print(f"Max states per run: {max_states_per_run}")

    df_rollout = generate_dataset_for_scenarios(
        scenarios=scenarios,
        behavior_heuristics=behavior_heuristics,
        candidate_heuristics=candidate_heuristics,
        rollout_preempt=False,
        max_states_per_run=max_states_per_run,
    )

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

    df_rollout.to_csv(f"{output_prefix}.csv", index=False)
    df_informative_with_ties.to_csv(f"{output_prefix}_informative_with_ties.csv", index=False)
    df_informative_no_ties.to_csv(f"{output_prefix}_informative_no_ties.csv", index=False)

    print("\n=== Full rollout dataset ===")
    print(df_rollout.shape)
    print(df_rollout["winner"].value_counts())

    print("\n=== Informative states, with ties ===")
    print(df_informative_with_ties.shape)
    print(df_informative_with_ties["winner"].value_counts())

    print("\n=== Informative states, no ties ===")
    print(df_informative_no_ties.shape)
    print(df_informative_no_ties["winner"].value_counts())

    print("\nSaved files:")
    print(f"- {output_prefix}.csv")
    print(f"- {output_prefix}_informative_with_ties.csv")
    print(f"- {output_prefix}_informative_no_ties.csv")

    return df_rollout, df_informative_with_ties, df_informative_no_ties


def main() -> None:
    build_rollout_dataset(
        include_canonical=True,
        include_stochastic=True,
        max_states_per_run=50,
        output_prefix="rollout_labeled_dataset_v2",
    )


if __name__ == "__main__":
    main()
