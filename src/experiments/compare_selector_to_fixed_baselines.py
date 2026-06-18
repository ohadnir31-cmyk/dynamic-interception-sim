
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HEURISTICS = ["NI", "FNI", "MPS", "Danger", "Ratio", "Cluster"]


def load_predictions(selector_dir: Path) -> pd.DataFrame:
    path = selector_dir / "selector_evaluation_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}\n"
            "Run evaluate_selector_regret.py first, preferably with --exclude-scenario-features."
        )

    print(f"Loading: {path}")
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"The file is empty: {path}")

    return df


def available_heuristics(df: pd.DataFrame) -> list[str]:
    found = []

    for h in HEURISTICS:
        required = [
            f"{h}_future_intercepted",
            f"{h}_regret",
            f"{h}_rank",
        ]
        if all(col in df.columns for col in required):
            found.append(h)

    if not found:
        raise ValueError(
            "No heuristic columns found. Expected columns such as "
            "NI_future_intercepted, NI_regret, NI_rank."
        )

    return found


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_policy_rows(
    source_df: pd.DataFrame,
    policy: str,
    policy_type: str,
    future_intercepted: pd.Series,
    regret: pd.Series,
    rank: pd.Series,
    future_escaped: pd.Series | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=source_df.index)

    metadata_cols = [
        "scenario",
        "t",
        "N_active",
        "N_active_bucket",
        "winner",
        "predicted_heuristic",
        "best_future_intercepted",
        "best_future_escaped",
    ]

    for col in metadata_cols:
        if col in source_df.columns:
            out[col] = source_df[col]

    out["policy"] = policy
    out["policy_type"] = policy_type
    out["future_intercepted"] = future_intercepted.to_numpy()
    out["regret"] = regret.to_numpy()
    out["rank"] = rank.to_numpy()

    if future_escaped is not None:
        out["future_escaped"] = future_escaped.to_numpy()

    out["zero_regret"] = out["regret"] == 0
    out["top1"] = out["rank"] <= 1
    out["top2"] = out["rank"] <= 2
    out["top3"] = out["rank"] <= 3

    return out.reset_index(drop=True)


def build_long_table(df: pd.DataFrame, heuristics: list[str]) -> tuple[pd.DataFrame, list[str]]:
    require_columns(df, ["best_future_intercepted"])

    parts = []

    oracle_future_escaped = (
        df["best_future_escaped"] if "best_future_escaped" in df.columns else None
    )

    parts.append(
        add_policy_rows(
            source_df=df,
            policy="Oracle",
            policy_type="oracle",
            future_intercepted=df["best_future_intercepted"],
            regret=pd.Series(np.zeros(len(df)), index=df.index),
            rank=pd.Series(np.ones(len(df)), index=df.index),
            future_escaped=oracle_future_escaped,
        )
    )

    require_columns(df, ["model_future_intercepted", "model_regret", "model_rank"])

    model_future_escaped = (
        df["model_future_escaped"] if "model_future_escaped" in df.columns else None
    )

    parts.append(
        add_policy_rows(
            source_df=df,
            policy="Learned selector",
            policy_type="learned",
            future_intercepted=df["model_future_intercepted"],
            regret=df["model_regret"],
            rank=df["model_rank"],
            future_escaped=model_future_escaped,
        )
    )

    for h in heuristics:
        require_columns(
            df,
            [f"{h}_future_intercepted", f"{h}_regret", f"{h}_rank"],
        )

        future_escaped = (
            df[f"{h}_future_escaped"] if f"{h}_future_escaped" in df.columns else None
        )

        parts.append(
            add_policy_rows(
                source_df=df,
                policy=f"Always {h}",
                policy_type="fixed",
                future_intercepted=df[f"{h}_future_intercepted"],
                regret=df[f"{h}_regret"],
                rank=df[f"{h}_rank"],
                future_escaped=future_escaped,
            )
        )

    long_df = pd.concat(parts, ignore_index=True)

    policy_order = ["Oracle", "Learned selector"] + [f"Always {h}" for h in heuristics]
    order_map = {policy: i for i, policy in enumerate(policy_order)}
    long_df["policy_order"] = long_df["policy"].map(order_map)

    long_df = long_df.sort_values(["policy_order", "scenario"]).reset_index(drop=True)

    return long_df, policy_order


def summarize_policies(long_df: pd.DataFrame, policy_order: list[str]) -> pd.DataFrame:
    summary = (
        long_df
        .groupby(["policy", "policy_type"], as_index=False)
        .agg(
            num_states=("future_intercepted", "size"),
            mean_future_intercepted=("future_intercepted", "mean"),
            median_future_intercepted=("future_intercepted", "median"),
            std_future_intercepted=("future_intercepted", "std"),
            mean_regret=("regret", "mean"),
            median_regret=("regret", "median"),
            max_regret=("regret", "max"),
            zero_regret_rate=("zero_regret", "mean"),
            top1_rate=("top1", "mean"),
            top2_rate=("top2", "mean"),
            top3_rate=("top3", "mean"),
        )
    )

    if "future_escaped" in long_df.columns:
        escaped = (
            long_df
            .groupby(["policy", "policy_type"], as_index=False)
            .agg(
                mean_future_escaped=("future_escaped", "mean"),
                median_future_escaped=("future_escaped", "median"),
            )
        )
        summary = summary.merge(escaped, on=["policy", "policy_type"], how="left")

    summary = summary[summary["num_states"] > 0].copy()

    oracle_rows = summary[summary["policy"] == "Oracle"]
    if oracle_rows.empty:
        raise ValueError("Oracle row is missing from summary.")

    oracle_mean = float(oracle_rows["mean_future_intercepted"].iloc[0])

    summary["gap_from_oracle"] = oracle_mean - summary["mean_future_intercepted"]
    summary["oracle_performance_ratio"] = summary["mean_future_intercepted"] / oracle_mean

    order_map = {policy: i for i, policy in enumerate(policy_order)}
    summary["policy_order"] = summary["policy"].map(order_map)

    summary = summary.sort_values(
        ["mean_future_intercepted", "zero_regret_rate"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return summary


def summarize_by_bucket(long_df: pd.DataFrame, policy_order: list[str]) -> pd.DataFrame:
    if "N_active_bucket" not in long_df.columns:
        return pd.DataFrame()

    out = (
        long_df
        .groupby(["N_active_bucket", "policy", "policy_type"], as_index=False)
        .agg(
            num_states=("future_intercepted", "size"),
            mean_future_intercepted=("future_intercepted", "mean"),
            median_future_intercepted=("future_intercepted", "median"),
            mean_regret=("regret", "mean"),
            median_regret=("regret", "median"),
            zero_regret_rate=("zero_regret", "mean"),
            top1_rate=("top1", "mean"),
            top2_rate=("top2", "mean"),
            top3_rate=("top3", "mean"),
        )
    )

    out = out[out["num_states"] > 0].copy()

    bucket_order = ["1", "2-3", "4-6", "7-10", "11+"]
    bucket_map = {bucket: i for i, bucket in enumerate(bucket_order)}
    policy_map = {policy: i for i, policy in enumerate(policy_order)}

    out["bucket_order"] = out["N_active_bucket"].map(bucket_map).fillna(999)
    out["policy_order"] = out["policy"].map(policy_map).fillna(999)

    out = out.sort_values(
        ["bucket_order", "mean_future_intercepted"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return out


def plot_bar(summary: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path) -> None:
    plot_df = summary.copy()
    plot_df = plot_df.sort_values(metric, ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["policy"], plot_df[metric])
    plt.xlabel("Policy")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {path}")


def save_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_bar(
        summary,
        "mean_future_intercepted",
        "Mean future interceptions by policy",
        "Mean future intercepted",
        output_dir / "fixed_baseline_mean_future_intercepted.png",
    )
    plot_bar(
        summary,
        "mean_regret",
        "Mean regret by policy",
        "Mean regret",
        output_dir / "fixed_baseline_mean_regret.png",
    )
    plot_bar(
        summary,
        "zero_regret_rate",
        "Zero-regret rate by policy",
        "Zero-regret rate",
        output_dir / "fixed_baseline_zero_regret_rate.png",
    )
    plot_bar(
        summary,
        "top3_rate",
        "Top-3 rate by policy",
        "Top-3 rate",
        output_dir / "fixed_baseline_top3_rate.png",
    )


def print_key_comparison(summary: pd.DataFrame) -> None:
    oracle = summary[summary["policy"] == "Oracle"].iloc[0]
    learned = summary[summary["policy"] == "Learned selector"].iloc[0]
    fixed = summary[summary["policy_type"] == "fixed"].copy()
    best_fixed = fixed.sort_values("mean_future_intercepted", ascending=False).iloc[0]

    diff_vs_best_fixed = (
        learned["mean_future_intercepted"] - best_fixed["mean_future_intercepted"]
    )

    print("\nKey comparison")
    print("--------------")
    print(f"Oracle mean future intercepted: {oracle['mean_future_intercepted']:.3f}")
    print(f"Learned selector mean future intercepted: {learned['mean_future_intercepted']:.3f}")
    print(
        f"Best fixed policy: {best_fixed['policy']} "
        f"with mean future intercepted = {best_fixed['mean_future_intercepted']:.3f}"
    )
    print(f"Learned selector gap from oracle: {learned['gap_from_oracle']:.3f}")
    print(f"Best fixed gap from oracle: {best_fixed['gap_from_oracle']:.3f}")
    print(f"Learned selector minus best fixed: {diff_vs_best_fixed:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare learned selector with fixed heuristic baselines. "
            "Recommended use: run this on the observable-only selector directory."
        )
    )

    parser.add_argument(
        "--selector-dir",
        required=True,
        type=str,
        help=(
            "Directory containing selector_evaluation_test_predictions.csv. "
            "Recommended: selector_regret_evaluation_observable_only."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Output directory. Default: selector-dir/fixed_baseline_comparison",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selector_dir = Path(args.selector_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else selector_dir / "fixed_baseline_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Selector vs Fixed-Heuristic Baseline Comparison ===")
    print(f"Selector directory: {selector_dir}")
    print(f"Output directory:   {output_dir}")

    df = load_predictions(selector_dir)
    heuristics = available_heuristics(df)

    print(f"Rows loaded: {len(df)}")
    print(f"Heuristics found: {heuristics}")

    long_df, policy_order = build_long_table(df, heuristics)
    summary = summarize_policies(long_df, policy_order)
    bucket_summary = summarize_by_bucket(long_df, policy_order)

    long_df.to_csv(output_dir / "fixed_baseline_policy_state_values_long.csv", index=False)
    summary.to_csv(output_dir / "fixed_baseline_comparison_summary.csv", index=False)

    if not bucket_summary.empty:
        bucket_summary.to_csv(
            output_dir / "fixed_baseline_comparison_by_active_bucket.csv",
            index=False,
        )

    save_plots(summary, output_dir)
    print_key_comparison(summary)

    print("\nPolicy summary")
    print("--------------")
    print(summary.drop(columns=["policy_order"], errors="ignore").to_string(index=False))

    if not bucket_summary.empty:
        print("\nPolicy summary by active-target bucket")
        print("--------------------------------------")
        print(
            bucket_summary
            .drop(columns=["bucket_order", "policy_order"], errors="ignore")
            .to_string(index=False)
        )

    print("\nDone.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
