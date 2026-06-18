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
            f"Could not find {path}. Run evaluate_selector_regret.py first."
        )
    print(f"Loading: {path}")
    return pd.read_csv(path)


def available_heuristics(df: pd.DataFrame) -> list[str]:
    found = []
    for h in HEURISTICS:
        required = [f"{h}_future_intercepted", f"{h}_regret", f"{h}_rank"]
        if all(c in df.columns for c in required):
            found.append(h)
    if not found:
        raise ValueError("No heuristic columns found in selector prediction file.")
    return found


def add_policy(
    df: pd.DataFrame,
    policy: str,
    policy_type: str,
    future_intercepted: pd.Series,
    regret: pd.Series,
    rank: pd.Series,
    future_escaped: pd.Series | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame()

    for col in [
        "scenario",
        "t",
        "N_active",
        "N_active_bucket",
        "winner",
        "predicted_heuristic",
        "best_future_intercepted",
        "best_future_escaped",
    ]:
        if col in df.columns:
            out[col] = df[col]

    out["policy"] = policy
    out["policy_type"] = policy_type
    out["future_intercepted"] = future_intercepted
    out["regret"] = regret
    out["rank"] = rank

    if future_escaped is not None:
        out["future_escaped"] = future_escaped

    out["zero_regret"] = out["regret"] == 0
    out["top1"] = out["rank"] <= 1
    out["top2"] = out["rank"] <= 2
    out["top3"] = out["rank"] <= 3

    return out


def build_long_table(df: pd.DataFrame, heuristics: list[str]) -> pd.DataFrame:
    parts = []

    # Oracle: hindsight best heuristic in every state
    parts.append(
        add_policy(
            df=df,
            policy="Oracle",
            policy_type="oracle",
            future_intercepted=df["best_future_intercepted"],
            regret=pd.Series(np.zeros(len(df)), index=df.index),
            rank=pd.Series(np.ones(len(df)), index=df.index),
            future_escaped=df["best_future_escaped"]
            if "best_future_escaped" in df.columns
            else None,
        )
    )

    # Learned selector: the heuristic predicted by the model
    for col in ["model_future_intercepted", "model_regret", "model_rank"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    parts.append(
        add_policy(
            df=df,
            policy="Learned selector",
            policy_type="learned",
            future_intercepted=df["model_future_intercepted"],
            regret=df["model_regret"],
            rank=df["model_rank"],
            future_escaped=df["model_future_escaped"]
            if "model_future_escaped" in df.columns
            else None,
        )
    )

    # Fixed heuristic baselines
    for h in heuristics:
        parts.append(
            add_policy(
                df=df,
                policy=f"Always {h}",
                policy_type="fixed",
                future_intercepted=df[f"{h}_future_intercepted"],
                regret=df[f"{h}_regret"],
                rank=df[f"{h}_rank"],
                future_escaped=df[f"{h}_future_escaped"]
                if f"{h}_future_escaped" in df.columns
                else None,
            )
        )

    long_df = pd.concat(parts, ignore_index=True)
    order = ["Oracle", "Learned selector"] + [f"Always {h}" for h in heuristics]
    long_df["policy"] = pd.Categorical(long_df["policy"], categories=order, ordered=True)
    return long_df.sort_values(["policy", "scenario"]).reset_index(drop=True)


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_df.groupby(["policy", "policy_type"], observed=False)
        .agg(
            num_states=("future_intercepted", "size"),
            mean_future_intercepted=("future_intercepted", "mean"),
            median_future_intercepted=("future_intercepted", "median"),
            mean_regret=("regret", "mean"),
            median_regret=("regret", "median"),
            max_regret=("regret", "max"),
            zero_regret_rate=("zero_regret", "mean"),
            top1_rate=("top1", "mean"),
            top2_rate=("top2", "mean"),
            top3_rate=("top3", "mean"),
        )
        .reset_index()
    )

    if "future_escaped" in long_df.columns:
        escaped = (
            long_df.groupby(["policy", "policy_type"], observed=False)
            .agg(mean_future_escaped=("future_escaped", "mean"))
            .reset_index()
        )
        summary = summary.merge(escaped, on=["policy", "policy_type"], how="left")

    oracle_mean = summary.loc[
        summary["policy"].astype(str) == "Oracle", "mean_future_intercepted"
    ].iloc[0]

    summary["gap_from_oracle"] = oracle_mean - summary["mean_future_intercepted"]
    summary["oracle_performance_ratio"] = summary["mean_future_intercepted"] / oracle_mean

    return summary.sort_values("mean_future_intercepted", ascending=False).reset_index(drop=True)


def summarize_by_bucket(long_df: pd.DataFrame) -> pd.DataFrame:
    if "N_active_bucket" not in long_df.columns:
        return pd.DataFrame()

    out = (
        long_df.groupby(["N_active_bucket", "policy", "policy_type"], observed=False)
        .agg(
            num_states=("future_intercepted", "size"),
            mean_future_intercepted=("future_intercepted", "mean"),
            mean_regret=("regret", "mean"),
            zero_regret_rate=("zero_regret", "mean"),
            top1_rate=("top1", "mean"),
            top2_rate=("top2", "mean"),
            top3_rate=("top3", "mean"),
        )
        .reset_index()
    )

    bucket_order = ["1", "2-3", "4-6", "7-10", "11+"]
    out["N_active_bucket"] = pd.Categorical(
        out["N_active_bucket"], categories=bucket_order, ordered=True
    )

    return out.sort_values(
        ["N_active_bucket", "mean_future_intercepted"], ascending=[True, False]
    ).reset_index(drop=True)


def plot_bar(summary: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path) -> None:
    plot_df = summary.copy()
    plot_df["policy"] = plot_df["policy"].astype(str)

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["policy"], plot_df[metric])
    plt.xlabel("Policy")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


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
    oracle = summary[summary["policy"].astype(str) == "Oracle"].iloc[0]
    learned = summary[summary["policy"].astype(str) == "Learned selector"].iloc[0]
    fixed = summary[summary["policy_type"] == "fixed"]
    best_fixed = fixed.sort_values("mean_future_intercepted", ascending=False).iloc[0]

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
    print(
        "Learned selector minus best fixed: "
        f"{learned['mean_future_intercepted'] - best_fixed['mean_future_intercepted']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare learned selector to fixed heuristic baselines."
    )
    parser.add_argument(
        "--selector-dir",
        required=True,
        type=str,
        help="Directory containing selector_evaluation_test_predictions.csv.",
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

    long_df = build_long_table(df, heuristics)
    summary = summarize(long_df)
    bucket_summary = summarize_by_bucket(long_df)

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
    print(summary.to_string(index=False))

    if not bucket_summary.empty:
        print("\nPolicy summary by active-target bucket")
        print("--------------------------------------")
        print(bucket_summary.to_string(index=False))

    print("\nDone.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
