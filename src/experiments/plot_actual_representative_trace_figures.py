from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOUNDARY_X = 0.0


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_column(df: pd.DataFrame, candidates: List[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns found: {candidates}")
    return None


def load_selected_states(selected_dir: Path) -> pd.DataFrame:
    candidates = [
        selected_dir / "selected_representative_states.csv",
        selected_dir / "representative_heuristic_states.csv",
        selected_dir / "selected_states.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Could not find selected representative states file in {selected_dir}"
    )


def load_rollout_states(selected_dir: Path) -> pd.DataFrame:
    candidates = [
        selected_dir.parent / "large_scale_rollout_states.csv",
        selected_dir / "large_scale_rollout_states.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Could not find large_scale_rollout_states.csv near {selected_dir}"
    )


def load_scenario_params(selected_dir: Path) -> pd.DataFrame:
    candidates = [
        selected_dir.parent / "large_scale_scenario_params.csv",
        selected_dir / "large_scale_scenario_params.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Could not find large_scale_scenario_params.csv near {selected_dir}"
    )


def load_full_rollouts(selected_dir: Path) -> pd.DataFrame:
    candidates = [
        selected_dir.parent / "large_scale_full_heuristic_rollouts.csv",
        selected_dir / "large_scale_full_heuristic_rollouts.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Could not find large_scale_full_heuristic_rollouts.csv near {selected_dir}"
    )


def find_trace_file(selected_dir: Path) -> Path:
    candidates = [
        selected_dir / "actual_trace_data.csv",
        selected_dir / "representative_trace_data.csv",
        selected_dir / "trace_data.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find trace data file in {selected_dir}. "
        f"Expected one of: {[str(c.name) for c in candidates]}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-dir",
        type=str,
        required=True,
        help="Directory containing selected representative states and trace data",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="actual_trace_figures",
        help="Subdirectory name for saved figures",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=4,
        help="Maximum number of frames to show per figure",
    )
    parser.add_argument(
        "--panels-per-row",
        type=int,
        default=2,
        help="Number of panels per row. Recommended: 2",
    )
    return parser.parse_args()


def get_unique_sorted_times(df: pd.DataFrame, time_col: str) -> List[float]:
    vals = sorted(df[time_col].dropna().unique().tolist())
    return vals


def choose_frame_times(
    trace_df: pd.DataFrame,
    time_col: str,
    max_frames: int,
) -> List[float]:
    times = get_unique_sorted_times(trace_df, time_col)
    if len(times) <= max_frames:
        return times
    idxs = np.linspace(0, len(times) - 1, max_frames).round().astype(int)
    idxs = sorted(set(idxs.tolist()))
    return [times[i] for i in idxs]


def get_axis_columns(trace_df: pd.DataFrame):
    scenario_col = pick_column(trace_df, ["scenario"])
    heuristic_col = pick_column(trace_df, ["heuristic"])
    time_col = pick_column(trace_df, ["t", "time"])
    obj_type_col = pick_column(trace_df, ["obj_type", "entity_type", "type"])
    x_col = pick_column(trace_df, ["x"])
    y_col = pick_column(trace_df, ["y"])
    target_id_col = pick_column(trace_df, ["target_id", "id"], required=False)
    selected_col = pick_column(
        trace_df,
        ["selected", "is_selected", "chosen", "selected_target"],
        required=False,
    )
    intercepted_col = pick_column(
        trace_df,
        ["intercepted", "is_intercepted"],
        required=False,
    )
    penetrated_col = pick_column(
        trace_df,
        ["penetrated", "escaped", "crossed_boundary"],
        required=False,
    )
    return {
        "scenario": scenario_col,
        "heuristic": heuristic_col,
        "time": time_col,
        "obj_type": obj_type_col,
        "x": x_col,
        "y": y_col,
        "target_id": target_id_col,
        "selected": selected_col,
        "intercepted": intercepted_col,
        "penetrated": penetrated_col,
    }


def split_frame_entities(frame_df: pd.DataFrame, cols: Dict[str, str]):
    obj_type_col = cols["obj_type"]
    if obj_type_col is None:
        raise ValueError("Trace file must contain obj_type/entity_type/type column")

    obj_type = frame_df[obj_type_col].astype(str).str.lower()
    interceptor_df = frame_df[obj_type.str.contains("interceptor|agent")]
    target_df = frame_df[obj_type.str.contains("target")]

    return interceptor_df, target_df


def compute_zoom_window(
    trace_df: pd.DataFrame,
    frame_times: List[float],
    cols: Dict[str, str],
) -> Tuple[float, float, float, float]:
    """
    Dynamic zoom that emphasizes the relevant interception region.
    Keeps only a very small margin to x<0 unless the interceptor/targets
    actually go there in the selected frames.
    """
    time_col = cols["time"]
    x_col = cols["x"]
    y_col = cols["y"]
    obj_type_col = cols["obj_type"]

    shown = trace_df[trace_df[time_col].isin(frame_times)].copy()

    if shown.empty:
        all_x = trace_df[x_col].to_numpy()
        all_y = trace_df[y_col].to_numpy()
    else:
        all_x = shown[x_col].to_numpy()
        all_y = shown[y_col].to_numpy()

    # Include the boundary x=0 in the view
    xmin_data = np.nanmin(all_x)
    xmax_data = np.nanmax(all_x)
    ymin_data = np.nanmin(all_y)
    ymax_data = np.nanmax(all_y)

    # Restrict how much empty x<0 space we show
    # If things slightly cross left of boundary, keep a small margin only.
    xmin = min(-1.0, xmin_data)
    xmin = max(xmin, -3.0)

    xmax = xmax_data
    ymin = ymin_data
    ymax = ymax_data

    x_range = max(1.0, xmax - xmin)
    y_range = max(1.0, ymax - ymin)

    x_pad = max(0.8, 0.08 * x_range)
    y_pad = max(0.8, 0.08 * y_range)

    xmin -= x_pad
    xmax += x_pad
    ymin -= y_pad
    ymax += y_pad

    # Keep aspect equal, but do not over-expand left side unnecessarily
    width = xmax - xmin
    height = ymax - ymin

    if width > height:
        extra = (width - height) / 2.0
        ymin -= extra
        ymax += extra
    else:
        extra = (height - width) / 2.0
        xmin -= extra
        xmax += extra

    # After square-adjustment, still cap the far-left space
    if xmin < -5:
        shift = -5 - xmin
        xmin += shift
        xmax += shift

    return xmin, xmax, ymin, ymax


def plot_boundary(ax):
    ax.axvline(BOUNDARY_X, color="black", linestyle="--", linewidth=1.5, label="Boundary")


def plot_targets(ax, target_df: pd.DataFrame, cols: Dict[str, str]):
    x_col = cols["x"]
    y_col = cols["y"]
    target_id_col = cols["target_id"]
    selected_col = cols["selected"]
    intercepted_col = cols["intercepted"]
    penetrated_col = cols["penetrated"]

    if target_df.empty:
        return

    target_df = target_df.copy()

    if selected_col is not None and selected_col in target_df.columns:
        selected_mask = target_df[selected_col].fillna(False).astype(bool)
    else:
        selected_mask = np.zeros(len(target_df), dtype=bool)

    if intercepted_col is not None and intercepted_col in target_df.columns:
        intercepted_mask = target_df[intercepted_col].fillna(False).astype(bool)
    else:
        intercepted_mask = np.zeros(len(target_df), dtype=bool)

    if penetrated_col is not None and penetrated_col in target_df.columns:
        penetrated_mask = target_df[penetrated_col].fillna(False).astype(bool)
    else:
        penetrated_mask = np.zeros(len(target_df), dtype=bool)

    normal_mask = ~(selected_mask | intercepted_mask | penetrated_mask)

    # Regular active targets
    if normal_mask.any():
        ax.scatter(
            target_df.loc[normal_mask, x_col],
            target_df.loc[normal_mask, y_col],
            s=55,
            c="tab:blue",
            marker="o",
            label="Active target",
            zorder=3,
        )

    # Selected target
    if selected_mask.any():
        ax.scatter(
            target_df.loc[selected_mask, x_col],
            target_df.loc[selected_mask, y_col],
            s=90,
            c="tab:red",
            marker="*",
            label="Selected target",
            zorder=5,
        )

    # Intercepted targets
    if intercepted_mask.any():
        ax.scatter(
            target_df.loc[intercepted_mask, x_col],
            target_df.loc[intercepted_mask, y_col],
            s=70,
            c="tab:green",
            marker="X",
            label="Intercepted target",
            zorder=4,
        )

    # Penetrated targets
    if penetrated_mask.any():
        ax.scatter(
            target_df.loc[penetrated_mask, x_col],
            target_df.loc[penetrated_mask, y_col],
            s=70,
            c="tab:gray",
            marker="x",
            label="Penetrated target",
            zorder=4,
        )

    # Optional small target-id labels
    if target_id_col is not None and target_id_col in target_df.columns:
        for _, row in target_df.iterrows():
            ax.text(
                row[x_col] + 0.10,
                row[y_col] + 0.10,
                str(int(row[target_id_col])) if pd.notna(row[target_id_col]) else "",
                fontsize=7,
                alpha=0.75,
                zorder=6,
            )


def plot_interceptor(ax, interceptor_df: pd.DataFrame, cols: Dict[str, str]):
    x_col = cols["x"]
    y_col = cols["y"]

    if interceptor_df.empty:
        return

    row = interceptor_df.iloc[0]
    ax.scatter(
        [row[x_col]],
        [row[y_col]],
        s=140,
        c="orange",
        marker="^",
        edgecolors="black",
        linewidths=0.8,
        label="Interceptor",
        zorder=6,
    )


def plot_interceptor_path(ax, trace_df: pd.DataFrame, cols: Dict[str, str]):
    time_col = cols["time"]
    x_col = cols["x"]
    y_col = cols["y"]
    obj_type_col = cols["obj_type"]

    obj_type = trace_df[obj_type_col].astype(str).str.lower()
    interceptor_trace = trace_df[obj_type.str.contains("interceptor|agent")].copy()

    if interceptor_trace.empty:
        return

    interceptor_trace = interceptor_trace.sort_values(time_col)
    ax.plot(
        interceptor_trace[x_col],
        interceptor_trace[y_col],
        color="orange",
        linewidth=1.6,
        alpha=0.9,
        label="Interceptor path",
        zorder=2,
    )


def get_summary_text(
    frame_df: pd.DataFrame,
    heuristic: str,
    t: float,
    cols: Dict[str, str],
) -> str:
    obj_type_col = cols["obj_type"]
    selected_col = cols["selected"]
    intercepted_col = cols["intercepted"]
    penetrated_col = cols["penetrated"]

    obj_type = frame_df[obj_type_col].astype(str).str.lower()
    targets = frame_df[obj_type.str.contains("target")].copy()

    n_active = len(targets)

    n_selected = 0
    if selected_col is not None and selected_col in targets.columns:
        n_selected = int(targets[selected_col].fillna(False).astype(bool).sum())

    n_intercepted = 0
    if intercepted_col is not None and intercepted_col in targets.columns:
        n_intercepted = int(targets[intercepted_col].fillna(False).astype(bool).sum())

    n_penetrated = 0
    if penetrated_col is not None and penetrated_col in targets.columns:
        n_penetrated = int(targets[penetrated_col].fillna(False).astype(bool).sum())

    text = (
        f"Heuristic: {heuristic}\n"
        f"t = {t:.2f}\n"
        f"Targets shown: {n_active}\n"
        f"Selected now: {n_selected}\n"
        f"Intercepted so far: {n_intercepted}\n"
        f"Penetrated so far: {n_penetrated}"
    )
    return text


def draw_single_panel(
    ax,
    trace_df: pd.DataFrame,
    frame_time: float,
    heuristic: str,
    cols: Dict[str, str],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
):
    time_col = cols["time"]

    frame_df = trace_df[np.isclose(trace_df[time_col], frame_time)].copy()
    interceptor_df, target_df = split_frame_entities(frame_df, cols)

    plot_boundary(ax)
    plot_interceptor_path(ax, trace_df[trace_df[time_col] <= frame_time], cols)
    plot_targets(ax, target_df, cols)
    plot_interceptor(ax, interceptor_df, cols)

    ax.set_title(f"{heuristic} - t={frame_time:.2f}", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

    # Information box on the RIGHT TOP, not left
    txt = get_summary_text(frame_df, heuristic, frame_time, cols)
    ax.text(
        1.02,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.7"),
        clip_on=False,
        zorder=10,
    )

    # Separate legend for each subplot, outside the plot on the right
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0.0,
        frameon=True,
        fontsize=8,
    )


def build_figure(
    scenario: str,
    heuristic: str,
    trace_df: pd.DataFrame,
    output_path: Path,
    max_frames: int,
    panels_per_row: int,
):
    cols = get_axis_columns(trace_df)
    time_col = cols["time"]

    trace_df = trace_df.sort_values(time_col).copy()
    frame_times = choose_frame_times(trace_df, time_col, max_frames=max_frames)

    if len(frame_times) == 0:
        return

    xmin, xmax, ymin, ymax = compute_zoom_window(trace_df, frame_times, cols)

    n_panels = len(frame_times)
    ncols = min(panels_per_row, n_panels)
    nrows = math.ceil(n_panels / ncols)

    # Wider figure because legend + text box are outside each subplot
    fig_w = 9.0 * ncols
    fig_h = 6.2 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, t in zip(axes, frame_times):
        draw_single_panel(
            ax=ax,
            trace_df=trace_df,
            frame_time=t,
            heuristic=heuristic,
            cols=cols,
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
        )

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"Representative trace - {heuristic} - {scenario}",
        fontsize=14,
        y=0.98,
    )

    # Important: more horizontal room for per-axis legend/info boxes
    plt.subplots_adjust(
        left=0.06,
        right=0.86,
        top=0.90,
        bottom=0.06,
        wspace=0.65,
        hspace=0.35,
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    selected_dir = Path(args.selected_dir)
    output_dir = selected_dir / args.output_subdir
    ensure_dir(output_dir)

    selected_states = load_selected_states(selected_dir)
    trace_path = find_trace_file(selected_dir)
    trace_df = safe_read_csv(trace_path)

    scenario_col_sel = pick_column(selected_states, ["scenario"])
    selected_for_col = pick_column(
        selected_states,
        ["selected_for", "heuristic", "winner", "representative_for"],
    )
    runner_up_col = pick_column(selected_states, ["runner_up"], required=False)

    trace_cols = get_axis_columns(trace_df)
    trace_scenario_col = trace_cols["scenario"]
    trace_heuristic_col = trace_cols["heuristic"]

    summary_rows = []

    for _, row in selected_states.iterrows():
        scenario = row[scenario_col_sel]
        heuristic = row[selected_for_col]
        runner_up = row[runner_up_col] if runner_up_col and runner_up_col in row.index else ""

        this_trace = trace_df[
            (trace_df[trace_scenario_col] == scenario)
            & (trace_df[trace_heuristic_col] == heuristic)
        ].copy()

        if this_trace.empty:
            print(f"Skipping empty trace for scenario={scenario}, heuristic={heuristic}")
            continue

        fig_name = f"{heuristic}__{scenario}.png"
        fig_path = output_dir / fig_name

        build_figure(
            scenario=scenario,
            heuristic=heuristic,
            trace_df=this_trace,
            output_path=fig_path,
            max_frames=args.max_frames,
            panels_per_row=args.panels_per_row,
        )

        summary_rows.append(
            {
                "scenario": scenario,
                "selected_for": heuristic,
                "runner_up": runner_up,
                "figure_path": str(fig_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "actual_trace_figure_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Saved figures to: {output_dir}")
    print(f"Saved summary:   {summary_path}")


if __name__ == "__main__":
    main()
