from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HEURISTICS = ["NI", "FNI", "FMTTB", "MPS", "FCluster"]

DEFAULT_FEATURES_TO_SHOW = [
    "N_active",
    "N_active_bucket",
    "min_ttb",
    "mean_ttb",
    "min_tti",
    "mean_tti",
    "min_slack",
    "mean_slack",
    "min_positive_slack",
    "count_feasible",
    "feasible_ratio",
    "cluster_index",
    "spatial_spread_x",
    "spatial_spread_y",
    "spatial_dispersion",
]


def load_state_dataset(input_dir: Path, dataset_mode: str) -> pd.DataFrame:
    if dataset_mode == "no_ties":
        filename = "large_scale_rollout_states_informative_no_ties.csv"
    elif dataset_mode == "with_ties":
        filename = "large_scale_rollout_states_informative_with_ties.csv"
    else:
        filename = "large_scale_rollout_states.csv"

    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset: {path}")

    print(f"Loading: {path}")
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return df


def infer_heuristics(df: pd.DataFrame) -> list[str]:
    found = []

    for h in HEURISTICS:
        if f"{h}_future_intercepted" in df.columns:
            found.append(h)

    if not found:
        raise ValueError(
            "No heuristic future-interception columns found. "
            "Expected columns such as NI_future_intercepted."
        )

    return found


def ensure_winner_columns(df: pd.DataFrame, heuristics: list[str]) -> pd.DataFrame:
    df = df.copy()
    score_cols = [f"{h}_future_intercepted" for h in heuristics]

    if "best_future_intercepted" not in df.columns:
        df["best_future_intercepted"] = df[score_cols].max(axis=1)

    if "winner" not in df.columns:
        df["winner"] = df[score_cols].idxmax(axis=1).str.replace(
            "_future_intercepted",
            "",
            regex=False,
        )

    score_values = df[score_cols].to_numpy(dtype=float)
    sorted_scores = np.sort(score_values, axis=1)

    best = sorted_scores[:, -1]
    second_best = sorted_scores[:, -2] if len(heuristics) >= 2 else sorted_scores[:, -1]

    df["diagnostic_best"] = best
    df["diagnostic_second_best"] = second_best
    df["winner_margin"] = best - second_best

    best_counts = (df[score_cols].eq(df["best_future_intercepted"], axis=0)).sum(axis=1)
    df["num_best_heuristics"] = best_counts

    return df


def select_representative_states(
    df: pd.DataFrame,
    heuristics: list[str],
    min_active: int,
    prefer_unique: bool,
) -> pd.DataFrame:
    selected_rows = []

    for h in heuristics:
        candidates = df[df["winner"] == h].copy()

        if min_active is not None:
            candidates = candidates[candidates["N_active"] >= min_active]

        if prefer_unique:
            unique_candidates = candidates[candidates["num_best_heuristics"] == 1].copy()
            if not unique_candidates.empty:
                candidates = unique_candidates

        if candidates.empty:
            print(f"No representative state found for {h}.")
            continue

        # Prefer cases where this heuristic is clearly better than the runner-up.
        # If margins tie, prefer more active targets and later larger best score.
        sort_cols = ["winner_margin", "N_active", "best_future_intercepted"]
        sort_cols = [c for c in sort_cols if c in candidates.columns]

        candidates = candidates.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        chosen = candidates.iloc[0].copy()
        chosen["selected_for"] = h
        selected_rows.append(chosen)

    if not selected_rows:
        raise ValueError("No representative states were selected.")

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    return selected


def build_selected_performance_table(
    selected: pd.DataFrame,
    heuristics: list[str],
) -> pd.DataFrame:
    rows = []

    for _, row in selected.iterrows():
        selected_for = row["selected_for"]

        for h in heuristics:
            perf_col = f"{h}_future_intercepted"
            regret_col = f"{h}_regret"
            rank_col = f"{h}_rank"
            escaped_col = f"{h}_future_escaped"

            rows.append(
                {
                    "selected_for": selected_for,
                    "heuristic": h,
                    "future_intercepted": row[perf_col] if perf_col in row else np.nan,
                    "regret": row[regret_col] if regret_col in row else np.nan,
                    "rank": row[rank_col] if rank_col in row else np.nan,
                    "future_escaped": row[escaped_col] if escaped_col in row else np.nan,
                }
            )

    return pd.DataFrame(rows)


def state_title(row: pd.Series) -> str:
    parts = [f"Representative state for {row['selected_for']}"]

    if "scenario" in row:
        parts.append(str(row["scenario"]))

    if "t" in row:
        try:
            parts.append(f"t={float(row['t']):.2f}")
        except Exception:
            parts.append(f"t={row['t']}")

    return "\n".join(parts)


def state_subtitle(row: pd.Series) -> str:
    bits = []

    for col in ["N_active", "N_active_bucket", "winner_margin"]:
        if col in row and pd.notna(row[col]):
            value = row[col]
            if isinstance(value, float):
                bits.append(f"{col}={value:.2f}")
            else:
                bits.append(f"{col}={value}")

    for col in ["min_ttb", "min_tti", "min_slack", "cluster_index", "spatial_dispersion"]:
        if col in row and pd.notna(row[col]):
            try:
                bits.append(f"{col}={float(row[col]):.2f}")
            except Exception:
                bits.append(f"{col}={row[col]}")

    return " | ".join(bits)


def plot_representative_state(
    row: pd.Series,
    heuristics: list[str],
    output_dir: Path,
) -> Path:
    selected_for = row["selected_for"]
    values = [row[f"{h}_future_intercepted"] for h in heuristics]

    plt.figure(figsize=(9, 5))
    plt.bar(heuristics, values)
    plt.xlabel("Heuristic")
    plt.ylabel("Future intercepted targets")
    plt.title(state_title(row))

    subtitle = state_subtitle(row)
    if subtitle:
        plt.suptitle(subtitle, y=0.02, fontsize=8)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    safe_scenario = str(row.get("scenario", "unknown")).replace("/", "_").replace("\\", "_")
    path = output_dir / f"representative_state_{selected_for}_{safe_scenario}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path


def plot_all_representative_states(
    selected: pd.DataFrame,
    heuristics: list[str],
    output_dir: Path,
) -> list[Path]:
    paths = []

    for _, row in selected.iterrows():
        path = plot_representative_state(row, heuristics, output_dir)
        paths.append(path)
        print(f"Saved figure: {path}")

    return paths


def save_feature_summary(
    selected: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    available_cols = [
        "selected_for",
        "scenario",
        "t",
        "winner",
        "winner_margin",
        "best_future_intercepted",
        "diagnostic_second_best",
        "num_best_heuristics",
    ]

    available_cols += [c for c in DEFAULT_FEATURES_TO_SHOW if c in selected.columns]
    available_cols = [c for c in available_cols if c in selected.columns]

    summary = selected[available_cols].copy()
    summary.to_csv(output_dir / "selected_representative_state_features.csv", index=False)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select representative diagnostic decision states where each heuristic "
            "is the rollout-based winner, and generate bar charts showing future "
            "interceptions by heuristic."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing rollout-state CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory for selected examples and figures.",
    )

    parser.add_argument(
        "--dataset-mode",
        choices=["no_ties", "with_ties", "full"],
        default="no_ties",
        help="Dataset to use. Default: no_ties.",
    )

    parser.add_argument(
        "--min-active",
        type=int,
        default=2,
        help="Minimum number of active targets for selected examples. Default: 2.",
    )

    parser.add_argument(
        "--allow-tied-best",
        action="store_true",
        help="Allow selected examples where the winner is tied with another heuristic.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Representative Heuristic Diagnostic States ===")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset mode:     {args.dataset_mode}")

    df = load_state_dataset(input_dir, args.dataset_mode)
    heuristics = infer_heuristics(df)
    df = ensure_winner_columns(df, heuristics)

    selected = select_representative_states(
        df=df,
        heuristics=heuristics,
        min_active=args.min_active,
        prefer_unique=not args.allow_tied_best,
    )

    selected.to_csv(output_dir / "selected_representative_states_full_rows.csv", index=False)

    feature_summary = save_feature_summary(selected, output_dir)
    perf_table = build_selected_performance_table(selected, heuristics)
    perf_table.to_csv(output_dir / "selected_representative_state_performance_long.csv", index=False)

    plot_all_representative_states(selected, heuristics, output_dir)

    print("\nSelected representative states")
    print("------------------------------")
    print(feature_summary.to_string(index=False))

    print("\nPerformance table")
    print("-----------------")
    print(perf_table.to_string(index=False))

    print("\nDone.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
