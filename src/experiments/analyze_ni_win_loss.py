from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEURISTICS = ["NI", "FNI", "FMTTB", "MPS", "FCluster"]

FEATURE_COLS = [
    "N_active",
    "min_ttb",
    "mean_ttb",
    "std_ttb",
    "min_tti",
    "mean_tti",
    "std_tti",
    "min_slack",
    "mean_slack",
    "std_slack",
    "min_positive_slack",
    "count_feasible",
    "count_negative_slack",
    "feasible_ratio",
    "cluster_index",
    "spatial_spread_x",
    "spatial_spread_y",
    "spatial_dispersion",
]

GROUP_COLUMNS = [
    "N_active_bucket",
    "scenario_regime",
    "spatial_structure",
    "arrival_process",
    "deadline_pressure",
    "behavior_heuristic",
]


DATASET_FILES = {
    "no_ties": "large_scale_rollout_states_informative_no_ties.csv",
    "with_ties": "large_scale_rollout_states_informative_with_ties.csv",
    "all": "large_scale_rollout_states.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze when the NI heuristic wins or loses in a state-level "
            "rollout-label dataset."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing large_scale_rollout_states*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Default: <input-dir>/ni_win_loss_analysis.",
    )
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default="no_ties",
        choices=sorted(DATASET_FILES.keys()),
        help="Which state-level dataset to analyze.",
    )
    parser.add_argument(
        "--strong-loss-threshold",
        type=float,
        default=2.0,
        help=(
            "A state is counted as a strong NI loss if NI_regret is at least "
            "this value."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows, useful for quick smoke tests.",
    )
    return parser.parse_args()


def n_active_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    return "11+"


def load_dataset(input_dir: Path, dataset_mode: str, max_rows: int | None) -> pd.DataFrame:
    path = input_dir / DATASET_FILES[dataset_mode]
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()

    if "N_active_bucket" not in df.columns and "N_active" in df.columns:
        df["N_active_bucket"] = df["N_active"].apply(lambda x: n_active_bucket(int(x)))

    required = ["winner", "NI_regret", "NI_future_intercepted", "best_future_intercepted"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def add_ni_labels(df: pd.DataFrame, strong_loss_threshold: float) -> pd.DataFrame:
    out = df.copy()

    out["NI_best"] = out["NI_regret"].fillna(np.inf) <= 0
    out["NI_loss"] = out["NI_regret"].fillna(0) > 0
    out["NI_strong_loss"] = out["NI_regret"].fillna(0) >= strong_loss_threshold

    # In a no-ties dataset, winner == NI is a strict NI win. In with-ties/all,
    # NI_best is the more meaningful quantity.
    out["NI_strict_winner"] = out["winner"].astype(str) == "NI"

    best_alt_cols = [f"{h}_future_intercepted" for h in HEURISTICS if h != "NI"]
    existing_alt_cols = [c for c in best_alt_cols if c in out.columns]
    if existing_alt_cols:
        out["best_alternative_future_intercepted"] = out[existing_alt_cols].max(axis=1)
        out["NI_minus_best_alternative"] = (
            out["NI_future_intercepted"] - out["best_alternative_future_intercepted"]
        )
    else:
        out["best_alternative_future_intercepted"] = np.nan
        out["NI_minus_best_alternative"] = np.nan

    return out


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    row = {
        "rows": len(df),
        "NI_best_rate": float(df["NI_best"].mean()),
        "NI_loss_rate": float(df["NI_loss"].mean()),
        "NI_strong_loss_rate": float(df["NI_strong_loss"].mean()),
        "NI_strict_winner_rate": float(df["NI_strict_winner"].mean()),
        "mean_NI_regret": float(df["NI_regret"].mean()),
        "median_NI_regret": float(df["NI_regret"].median()),
        "max_NI_regret": float(df["NI_regret"].max()),
        "mean_best_future_intercepted": float(df["best_future_intercepted"].mean()),
        "mean_NI_future_intercepted": float(df["NI_future_intercepted"].mean()),
        "mean_NI_minus_best_alternative": float(df["NI_minus_best_alternative"].mean()),
    }
    return pd.DataFrame([row])


def dominant_winner_when_ni_loses(g: pd.DataFrame) -> str:
    loss = g[g["NI_loss"]]
    if loss.empty:
        return ""
    counts = loss["winner"].astype(str).value_counts()
    return str(counts.index[0])


def group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if group_col not in df.columns:
        return pd.DataFrame()

    for value, g in df.groupby(group_col, dropna=False):
        if len(g) == 0:
            continue
        rows.append(
            {
                group_col: value,
                "rows": len(g),
                "row_share": float(len(g) / max(1, len(df))),
                "NI_best_rate": float(g["NI_best"].mean()),
                "NI_loss_rate": float(g["NI_loss"].mean()),
                "NI_strong_loss_rate": float(g["NI_strong_loss"].mean()),
                "mean_NI_regret": float(g["NI_regret"].mean()),
                "median_NI_regret": float(g["NI_regret"].median()),
                "mean_best_future_intercepted": float(g["best_future_intercepted"].mean()),
                "mean_NI_future_intercepted": float(g["NI_future_intercepted"].mean()),
                "dominant_winner_when_NI_loses": dominant_winner_when_ni_loses(g),
                "mean_N_active": float(g["N_active"].mean()) if "N_active" in g.columns else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if group_col == "N_active_bucket":
        order = {"1": 0, "2-3": 1, "4-6": 2, "7-10": 3, "11+": 4}
        out["_order"] = out[group_col].map(order).fillna(999)
        out = out.sort_values(["_order", group_col]).drop(columns=["_order"])
    else:
        out = out.sort_values("NI_loss_rate", ascending=False)

    return out


def winner_distribution_when_ni_loses(df: pd.DataFrame) -> pd.DataFrame:
    loss = df[df["NI_loss"]].copy()
    if loss.empty:
        return pd.DataFrame(columns=["winner", "rows", "share", "mean_NI_regret"])

    rows = []
    for winner, g in loss.groupby("winner"):
        rows.append(
            {
                "winner": winner,
                "rows": len(g),
                "share": float(len(g) / len(loss)),
                "mean_NI_regret": float(g["NI_regret"].mean()),
                "median_NI_regret": float(g["NI_regret"].median()),
                "mean_best_future_intercepted": float(g["best_future_intercepted"].mean()),
                "mean_NI_future_intercepted": float(g["NI_future_intercepted"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def feature_comparison(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    existing = [c for c in feature_cols if c in df.columns]

    ni_best = df[df["NI_best"]]
    ni_loss = df[df["NI_loss"]]
    ni_strong_loss = df[df["NI_strong_loss"]]

    for col in existing:
        best_values = pd.to_numeric(ni_best[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        loss_values = pd.to_numeric(ni_loss[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        strong_values = pd.to_numeric(ni_strong_loss[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        if len(best_values) == 0 or len(loss_values) == 0:
            continue

        mean_best = float(best_values.mean())
        mean_loss = float(loss_values.mean())
        std_best = float(best_values.std(ddof=1)) if len(best_values) > 1 else 0.0
        std_loss = float(loss_values.std(ddof=1)) if len(loss_values) > 1 else 0.0
        pooled = float(np.sqrt((std_best * std_best + std_loss * std_loss) / 2.0))
        standardized_delta = (mean_loss - mean_best) / pooled if pooled > 1e-12 else np.nan

        rows.append(
            {
                "feature": col,
                "NI_best_mean": mean_best,
                "NI_loss_mean": mean_loss,
                "NI_loss_minus_best_mean": mean_loss - mean_best,
                "standardized_delta_loss_vs_best": standardized_delta,
                "NI_best_median": float(best_values.median()),
                "NI_loss_median": float(loss_values.median()),
                "NI_strong_loss_mean": float(strong_values.mean()) if len(strong_values) else np.nan,
                "NI_strong_loss_median": float(strong_values.median()) if len(strong_values) else np.nan,
                "best_rows": len(best_values),
                "loss_rows": len(loss_values),
                "strong_loss_rows": len(strong_values),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["abs_standardized_delta"] = out["standardized_delta_loss_vs_best"].abs()
    out = out.sort_values("abs_standardized_delta", ascending=False)
    return out


def pairwise_policy_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for heuristic in HEURISTICS:
        fi = f"{heuristic}_future_intercepted"
        fr = f"{heuristic}_regret"
        rank = f"{heuristic}_rank"
        if fi not in df.columns or fr not in df.columns:
            continue
        rows.append(
            {
                "heuristic": heuristic,
                "mean_future_intercepted": float(df[fi].mean()),
                "median_future_intercepted": float(df[fi].median()),
                "mean_regret": float(df[fr].mean()),
                "median_regret": float(df[fr].median()),
                "zero_regret_rate": float((df[fr] <= 0).mean()),
                "top1_rate": float((df[rank] <= 1).mean()) if rank in df.columns else np.nan,
                "top2_rate": float((df[rank] <= 2).mean()) if rank in df.columns else np.nan,
                "top3_rate": float((df[rank] <= 3).mean()) if rank in df.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_future_intercepted", ascending=False)


def save_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    rotate: bool = False,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df[x_col].astype(str), df[y_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(axis="y", alpha=0.25)
    if rotate:
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = summarize_overall(df)
    overall.to_csv(output_dir / "ni_win_loss_overall.csv", index=False)

    policy = pairwise_policy_summary(df)
    policy.to_csv(output_dir / "policy_summary.csv", index=False)

    winner_dist = winner_distribution_when_ni_loses(df)
    winner_dist.to_csv(output_dir / "winner_distribution_when_NI_loses.csv", index=False)

    features = feature_comparison(df, FEATURE_COLS)
    features.to_csv(output_dir / "ni_loss_feature_comparison.csv", index=False)

    for group_col in GROUP_COLUMNS:
        g = group_summary(df, group_col)
        if not g.empty:
            g.to_csv(output_dir / f"ni_loss_by_{group_col}.csv", index=False)

    # Compact figures for quick inspection.
    by_bucket = group_summary(df, "N_active_bucket")
    save_bar(
        by_bucket,
        x_col="N_active_bucket",
        y_col="NI_loss_rate",
        title="NI loss rate by active-target bucket",
        output_path=output_dir / "ni_loss_rate_by_active_bucket.png",
    )

    save_bar(
        by_bucket,
        x_col="N_active_bucket",
        y_col="mean_NI_regret",
        title="Mean NI regret by active-target bucket",
        output_path=output_dir / "mean_NI_regret_by_active_bucket.png",
    )

    save_bar(
        winner_dist,
        x_col="winner",
        y_col="rows",
        title="Which heuristic wins when NI loses",
        output_path=output_dir / "winner_distribution_when_NI_loses.png",
    )

    top_features = features.head(12).copy()
    if not top_features.empty:
        top_features = top_features.sort_values("standardized_delta_loss_vs_best")
        save_bar(
            top_features,
            x_col="feature",
            y_col="standardized_delta_loss_vs_best",
            title="Feature shift: NI-loss states vs NI-best states",
            output_path=output_dir / "ni_loss_feature_shift_top12.png",
            rotate=True,
        )

    # Save the row-level dataset with labels for optional notebook inspection.
    df.to_csv(output_dir / "ni_win_loss_labeled_states.csv", index=False)


def print_table(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if df.empty:
        print("<empty>")
    else:
        print(df.head(max_rows).to_string(index=False))


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "ni_win_loss_analysis"

    df = load_dataset(
        input_dir=input_dir,
        dataset_mode=args.dataset_mode,
        max_rows=args.max_rows,
    )
    df = add_ni_labels(df, strong_loss_threshold=args.strong_loss_threshold)

    save_outputs(df, output_dir)

    print("\n=== NI Win/Loss Analysis ===")
    print(f"Input dir:      {input_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Dataset mode:   {args.dataset_mode}")
    print(f"Rows analyzed:  {len(df)}")
    print(f"Strong loss threshold: NI_regret >= {args.strong_loss_threshold}")

    print_table("Overall", summarize_overall(df))
    print_table("Policy summary", pairwise_policy_summary(df))
    print_table("NI loss by active-target bucket", group_summary(df, "N_active_bucket"))
    print_table("Winner distribution when NI loses", winner_distribution_when_ni_loses(df))
    print_table("Top feature shifts: NI-loss vs NI-best", feature_comparison(df, FEATURE_COLS), max_rows=12)

    print("\nSaved outputs:")
    for p in sorted(output_dir.glob("*")):
        if p.is_file():
            print(p)


if __name__ == "__main__":
    main()
