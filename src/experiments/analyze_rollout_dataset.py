from __future__ import annotations

import pandas as pd


FEATURE_COLS = [
    "N_active",
    "min_ttb",
    "mean_ttb",
    "min_positive_slack",
    "mean_slack",
    "count_negative_slack",
    "mean_tti",
    "cluster_index",
]


def analyze_dataset(
    input_csv: str = "rollout_labeled_dataset_medium_informative_no_ties.csv",
    output_prefix: str = "rollout_analysis",
) -> None:
    df = pd.read_csv(input_csv)

    print("\n=== Dataset shape ===")
    print(df.shape)

    print("\n=== Winner distribution ===")
    print(df["winner"].value_counts())
    print("\n=== Winner distribution (%) ===")
    print((df["winner"].value_counts(normalize=True) * 100).round(2))

    print("\n=== Scenario distribution ===")
    print(df["scenario"].value_counts().head(20))

    print("\n=== Feature means by winner ===")
    feature_means = df.groupby("winner")[FEATURE_COLS].mean(numeric_only=True)
    print(feature_means.round(3))

    print("\n=== Feature medians by winner ===")
    feature_medians = df.groupby("winner")[FEATURE_COLS].median(numeric_only=True)
    print(feature_medians.round(3))

    print("\n=== Feature std by winner ===")
    feature_stds = df.groupby("winner")[FEATURE_COLS].std(numeric_only=True)
    print(feature_stds.round(3))

    print("\n=== Average N_active by winner ===")
    print(df.groupby("winner")["N_active"].mean().round(3))

    print("\n=== Average negative slack count by winner ===")
    print(df.groupby("winner")["count_negative_slack"].mean().round(3))

    feature_means.to_csv(f"{output_prefix}_feature_means_by_winner.csv")
    feature_medians.to_csv(f"{output_prefix}_feature_medians_by_winner.csv")
    feature_stds.to_csv(f"{output_prefix}_feature_stds_by_winner.csv")

    summary = df["winner"].value_counts().reset_index()
    summary.columns = ["winner", "count"]
    summary["percent"] = 100 * summary["count"] / summary["count"].sum()
    summary.to_csv(f"{output_prefix}_winner_distribution.csv", index=False)

    print("\nSaved:")
    print(f"- {output_prefix}_feature_means_by_winner.csv")
    print(f"- {output_prefix}_feature_medians_by_winner.csv")
    print(f"- {output_prefix}_feature_stds_by_winner.csv")
    print(f"- {output_prefix}_winner_distribution.csv")


def main() -> None:
    analyze_dataset(
        input_csv="rollout_labeled_dataset_medium_informative_no_ties.csv",
        output_prefix="rollout_analysis_medium",
    )


if __name__ == "__main__":
    main()
