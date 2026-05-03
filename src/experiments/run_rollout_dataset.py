from __future__ import annotations

import pandas as pd

from src.experiments.canonical_scenarios import get_canonical_scenarios
from src.experiments.rollout_labeling import generate_dataset_for_scenarios


def build_rollout_dataset(
    max_states_per_run: int | None = 10,
    output_prefix: str = "rollout_labeled_dataset_v1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = get_canonical_scenarios()

    candidate_heuristics = ["NI", "TTB", "MPS", "Weighted", "Cluster"]
    behavior_heuristics = ["NI", "MPS", "Cluster"]

    df_rollout = generate_dataset_for_scenarios(
        scenarios=scenarios,
        behavior_heuristics=behavior_heuristics,
        candidate_heuristics=candidate_heuristics,
        rollout_preempt=False,
        max_states_per_run=max_states_per_run,
    )

    df_informative = filter_informative_states(
        df_rollout=df_rollout,
        candidate_heuristics=candidate_heuristics,
    )

    df_rollout.to_csv(f"{output_prefix}.csv", index=False)
    df_informative.to_csv(f"{output_prefix}_informative.csv", index=False)

    print("\n=== Full rollout dataset ===")
    print(df_rollout.shape)
    print(df_rollout["winner"].value_counts())

    print("\n=== Winner sets ===")
    print(df_rollout["winner_set"].value_counts())

    print("\n=== Informative states only ===")
    print(df_informative.shape)
    if not df_informative.empty:
        print(df_informative["winner"].value_counts())
        print(df_informative["winner_set"].value_counts())

    print("\nSaved files:")
    print(f"- {output_prefix}.csv")
    print(f"- {output_prefix}_informative.csv")

    return df_rollout, df_informative


def filter_informative_states(
    df_rollout: pd.DataFrame,
    candidate_heuristics: list[str],
) -> pd.DataFrame:
    if df_rollout.empty:
        return df_rollout.copy()

    df = df_rollout[
        (df_rollout["N_active"] >= 2)
        & (df_rollout["n_winners"] < len(candidate_heuristics))
    ].copy()

    return df


def main() -> None:
    build_rollout_dataset(
        max_states_per_run=10,
        output_prefix="rollout_labeled_dataset_v1",
    )


if __name__ == "__main__":
    main()
