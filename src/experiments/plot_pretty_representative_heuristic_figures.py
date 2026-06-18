from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HEURISTIC_ORDER = ["NI", "FNI", "MPS", "Danger", "Ratio", "Cluster"]


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf_path = input_dir / "selected_representative_state_performance_long.csv"
    features_path = input_dir / "selected_representative_state_features.csv"

    if not perf_path.exists():
        raise FileNotFoundError(f"Missing file: {perf_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Missing file: {features_path}")

    perf = pd.read_csv(perf_path)
    features = pd.read_csv(features_path)

    return perf, features


def ordered_heuristics(perf: pd.DataFrame) -> list[str]:
    present = list(perf["heuristic"].dropna().unique())
    return [h for h in HEURISTIC_ORDER if h in present] + [
        h for h in present if h not in HEURISTIC_ORDER
    ]


def format_num(x, digits: int = 2) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)

    if np.isnan(x):
        return "NA"

    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))

    return f"{x:.{digits}f}"


def describe_state(row: pd.Series) -> str:
    parts = []

    if "scenario" in row and pd.notna(row["scenario"]):
        parts.append(str(row["scenario"]))

    if "t" in row and pd.notna(row["t"]):
        parts.append(f"t={format_num(row['t'])}")

    if "N_active" in row and pd.notna(row["N_active"]):
        parts.append(f"N={format_num(row['N_active'])}")

    if "N_active_bucket" in row and pd.notna(row["N_active_bucket"]):
        parts.append(f"bucket={row['N_active_bucket']}")

    if "winner_margin" in row and pd.notna(row["winner_margin"]):
        parts.append(f"margin={format_num(row['winner_margin'])}")

    return " | ".join(parts)


def describe_operational_features(row: pd.Series) -> str:
    feature_names = [
        ("min_ttb", "min TTB"),
        ("min_tti", "min TTI"),
        ("min_slack", "min slack"),
        ("feasible_ratio", "feasible ratio"),
        ("cluster_index", "cluster index"),
        ("spatial_dispersion", "dispersion"),
    ]

    parts = []

    for col, label in feature_names:
        if col in row and pd.notna(row[col]):
            parts.append(f"{label}: {format_num(row[col])}")

    return "   ".join(parts)


def add_value_labels(ax, values: list[float]) -> None:
    xmax = max(values) if values else 0
    offset = max(0.15, xmax * 0.015)

    for patch, value in zip(ax.patches, values):
        x = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(
            x + offset,
            y,
            format_num(value),
            va="center",
            ha="left",
            fontsize=9,
        )


def plot_single_example(
    perf: pd.DataFrame,
    features: pd.DataFrame,
    selected_for: str,
    heuristics: list[str],
    output_dir: Path,
) -> Path:
    df = perf[perf["selected_for"] == selected_for].copy()
    df["heuristic"] = pd.Categorical(df["heuristic"], categories=heuristics, ordered=True)
    df = df.sort_values("heuristic")

    feature_rows = features[features["selected_for"] == selected_for]
    feature_row = feature_rows.iloc[0] if not feature_rows.empty else pd.Series(dtype=object)

    values = df["future_intercepted"].astype(float).tolist()
    y_labels = df["heuristic"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    bars = ax.barh(y_labels, values)
    ax.invert_yaxis()

    for bar, h in zip(bars, y_labels):
        if h == selected_for:
            bar.set_linewidth(2.6)
            bar.set_edgecolor("black")

    ax.set_xlabel("Future intercepted targets")
    ax.set_ylabel("Heuristic")
    ax.set_title(f"Representative state where {selected_for} is best", fontsize=14, pad=14)

    subtitle = describe_state(feature_row)
    if subtitle:
        ax.text(
            0,
            1.04,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
        )

    feature_text = describe_operational_features(feature_row)
    if feature_text:
        ax.text(
            0,
            -0.20,
            feature_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
        )

    add_value_labels(ax, values)

    xmax = max(values) if values else 1
    ax.set_xlim(0, xmax * 1.18 + 0.5)
    ax.grid(axis="x", alpha=0.25)

    safe_name = selected_for.replace("/", "_")
    path = output_dir / f"pretty_representative_{safe_name}.png"

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_panel(
    perf: pd.DataFrame,
    features: pd.DataFrame,
    heuristics: list[str],
    selected_order: list[str],
    output_dir: Path,
) -> Path:
    n = len(selected_order)
    cols = 2
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.4 * rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, selected_for in zip(axes_flat, selected_order):
        df = perf[perf["selected_for"] == selected_for].copy()
        df["heuristic"] = pd.Categorical(df["heuristic"], categories=heuristics, ordered=True)
        df = df.sort_values("heuristic")

        values = df["future_intercepted"].astype(float).tolist()
        y_labels = df["heuristic"].astype(str).tolist()

        bars = ax.barh(y_labels, values)
        ax.invert_yaxis()

        for bar, h in zip(bars, y_labels):
            if h == selected_for:
                bar.set_linewidth(2.4)
                bar.set_edgecolor("black")

        feature_rows = features[features["selected_for"] == selected_for]
        feature_row = feature_rows.iloc[0] if not feature_rows.empty else pd.Series(dtype=object)

        ax.set_title(f"{selected_for} wins", fontsize=12)
        ax.set_xlabel("Future intercepted")
        ax.grid(axis="x", alpha=0.25)

        subtitle = describe_state(feature_row)
        if subtitle:
            ax.text(
                0,
                1.03,
                subtitle,
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
            )

        xmax = max(values) if values else 1
        ax.set_xlim(0, xmax * 1.20 + 0.5)

        for patch, value in zip(ax.patches, values):
            x = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            ax.text(
                x + max(0.1, xmax * 0.015),
                y,
                format_num(value),
                va="center",
                ha="left",
                fontsize=8,
            )

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        "Representative diagnostic states by winning heuristic",
        fontsize=16,
        y=1.01,
    )

    fig.tight_layout()
    path = output_dir / "pretty_representative_states_panel.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def create_caption_table(
    perf: pd.DataFrame,
    features: pd.DataFrame,
    selected_order: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []

    for selected_for in selected_order:
        df = perf[perf["selected_for"] == selected_for].copy()
        df = df.sort_values("future_intercepted", ascending=False)

        best = df.iloc[0]
        runner = df.iloc[1] if len(df) > 1 else None

        feature_rows = features[features["selected_for"] == selected_for]
        f = feature_rows.iloc[0] if not feature_rows.empty else pd.Series(dtype=object)

        rows.append(
            {
                "selected_for": selected_for,
                "scenario": f.get("scenario", ""),
                "t": f.get("t", ""),
                "N_active": f.get("N_active", ""),
                "N_active_bucket": f.get("N_active_bucket", ""),
                "winner_margin": f.get("winner_margin", ""),
                "best_intercepted": best["future_intercepted"],
                "runner_up": runner["heuristic"] if runner is not None else "",
                "runner_up_intercepted": runner["future_intercepted"] if runner is not None else "",
                "min_ttb": f.get("min_ttb", ""),
                "min_tti": f.get("min_tti", ""),
                "min_slack": f.get("min_slack", ""),
                "feasible_ratio": f.get("feasible_ratio", ""),
                "cluster_index": f.get("cluster_index", ""),
                "spatial_dispersion": f.get("spatial_dispersion", ""),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "pretty_representative_caption_table.csv", index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create polished publication-ready figures from the representative "
            "heuristic states selected by select_representative_heuristic_states.py."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing selected_representative_state_*.csv files.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Output directory. Default: input-dir/pretty_figures",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "pretty_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Pretty Representative Heuristic Figures ===")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    perf, features = load_inputs(input_dir)
    heuristics = ordered_heuristics(perf)
    selected_order = [h for h in heuristics if h in set(perf["selected_for"])]

    print(f"Heuristics: {heuristics}")
    print(f"Selected examples: {selected_order}")

    paths = []
    for selected_for in selected_order:
        path = plot_single_example(perf, features, selected_for, heuristics, output_dir)
        paths.append(path)
        print(f"Saved figure: {path}")

    panel_path = plot_panel(perf, features, heuristics, selected_order, output_dir)
    print(f"Saved panel:  {panel_path}")

    caption_table = create_caption_table(perf, features, selected_order, output_dir)
    print(f"Saved captions table: {output_dir / 'pretty_representative_caption_table.csv'}")

    print("\nCaption table")
    print("-------------")
    print(caption_table.to_string(index=False))

    print("\nDone.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
