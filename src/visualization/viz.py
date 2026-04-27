from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. Basic animation
# ============================================================

def animate_trace(
    trace,
    boundary_x: float = 0.0,
    xlim: Tuple[float, float] = (-5, 80),
    ylim: Tuple[float, float] = (-60, 60),
    interval: int = 300,
):
    """
    Basic animation of one simulation trace.
    Useful for debugging and qualitative inspection in Colab.
    """

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.axvline(
        boundary_x,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Boundary",
    )

    interceptor_plot, = ax.plot(
        [], [], "bo", markersize=8, label="Interceptor"
    )
    threats_plot, = ax.plot(
        [], [], "ro", linestyle="None", markersize=6, label="Threats"
    )
    chosen_plot, = ax.plot(
        [], [], "go", linestyle="None", markersize=10, label="Chosen target"
    )

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    stats = ax.text(0.02, 0.88, "", transform=ax.transAxes, va="top")

    ax.legend(loc="upper right")

    def update(frame_idx):
        state = trace[frame_idx]

        pI = state["interceptor_pos"]
        threats = state["active_threats"]
        chosen_id = state["chosen_target_id"]

        interceptor_plot.set_data([pI[0]], [pI[1]])

        if threats:
            xs = [th["pos"][0] for th in threats]
            ys = [th["pos"][1] for th in threats]
            threats_plot.set_data(xs, ys)

            chosen_x, chosen_y = [], []
            for th in threats:
                if th["id"] == chosen_id:
                    chosen_x.append(th["pos"][0])
                    chosen_y.append(th["pos"][1])

            chosen_plot.set_data(chosen_x, chosen_y)
        else:
            threats_plot.set_data([], [])
            chosen_plot.set_data([], [])

        title.set_text(f"t = {state['t']:.1f}")

        f = state["features"]
        stats.set_text(
            f"N_active={f['N_active']}\n"
            f"min_TTB={_fmt_num(f['min_ttb'])}\n"
            f"min_pos_slack={_fmt_num(f['min_positive_slack'])}\n"
            f"neg_slack={f['count_negative_slack']}\n"
            f"intercepted={state['intercepted_so_far']}\n"
            f"escaped={state['escaped_so_far']}"
        )

        return interceptor_plot, threats_plot, chosen_plot, title, stats

    ani = FuncAnimation(
        fig,
        update,
        frames=len(trace),
        interval=interval,
        blit=False,
    )

    plt.close(fig)
    return ani


# ============================================================
# 2. Static path plot
# ============================================================

def plot_static_paths(
    trace,
    boundary_x: float = 0.0,
    xlim: Tuple[float, float] = (-5, 80),
    ylim: Tuple[float, float] = (-60, 60),
):
    """
    Simple static path plot for one trace.
    Shows interceptor path and first/last seen threat positions.
    """

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.axvline(
        boundary_x,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Boundary",
    )

    interceptor_path = np.array([state["interceptor_pos"] for state in trace])
    ax.plot(
        interceptor_path[:, 0],
        interceptor_path[:, 1],
        "b-",
        linewidth=2,
        label="Interceptor path",
    )

    first_seen = {}
    last_seen = {}

    for state in trace:
        for th in state["active_threats"]:
            tid = th["id"]
            pos = np.array(th["pos"])

            if tid not in first_seen:
                first_seen[tid] = pos

            last_seen[tid] = pos

    if first_seen:
        spawn_x = [p[0] for p in first_seen.values()]
        spawn_y = [p[1] for p in first_seen.values()]
        ax.scatter(
            spawn_x,
            spawn_y,
            c="red",
            s=40,
            label="First seen positions",
        )

    if last_seen:
        end_x = [p[0] for p in last_seen.values()]
        end_y = [p[1] for p in last_seen.values()]
        ax.scatter(
            end_x,
            end_y,
            c="gray",
            s=20,
            label="Last seen positions",
        )

    ax.legend(loc="upper right")
    ax.set_title("Static path view")
    ax.grid(alpha=0.25)

    return fig, ax


# ============================================================
# 3. Publication-style frame grid
# ============================================================

def plot_policy_frames_publication(
    trace,
    policy_name: str = "NI",
    save_path: str = "policy_frames_publication.png",
    frame_indices: Optional[Sequence[int]] = None,
    n_cols: int = 2,
    xlim: Tuple[float, float] = (-5, 45),
    ylim: Tuple[float, float] = (-35, 35),
):
    """
    Publication-style grid of selected frames from one simulation trace.

    This is intended for:
    - Word reports
    - PowerPoint slides
    - research documentation

    It produces:
    - numbered frames
    - readable state box
    - clear legend
    - highlighted chosen target
    - arrow from interceptor to chosen target
    """

    if len(trace) == 0:
        raise ValueError("Trace is empty. Cannot plot frames.")

    if frame_indices is None:
        frame_indices = _default_frame_indices(trace, max_frames=8)

    selected = [trace[i] for i in frame_indices if 0 <= i < len(trace)]

    if len(selected) == 0:
        raise ValueError("No valid frame indices were provided.")

    n_frames = len(selected)
    n_rows = math.ceil(n_frames / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.2 * n_cols, 5.2 * n_rows),
        constrained_layout=True,
    )

    axes = np.array(axes).reshape(-1)

    for k, (state, ax) in enumerate(zip(selected, axes), start=1):
        _plot_single_publication_frame(
            ax=ax,
            state=state,
            frame_number=k,
            policy_name=policy_name,
            xlim=xlim,
            ylim=ylim,
            show_legend_labels=(k == 1),
        )

    # Hide unused axes
    for ax in axes[len(selected):]:
        ax.axis("off")

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        f"Simulation snapshots under {policy_name} heuristic",
        fontsize=16,
        fontweight="bold",
        y=1.04,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return save_path


# ============================================================
# 4. Side-by-side publication comparison
# ============================================================

def plot_two_policy_comparison_publication(
    trace_left,
    trace_right,
    left_name: str = "NI",
    right_name: str = "MPS",
    save_path: str = "policy_comparison_publication.png",
    frame_indices: Optional[Sequence[int]] = None,
    xlim: Tuple[float, float] = (-5, 45),
    ylim: Tuple[float, float] = (-35, 35),
):
    """
    Publication-style comparison between two policies on the same scenario.

    Each row shows the same selected time/frame:
    - left: policy A
    - right: policy B

    This is usually the best format for a report or presentation.
    """

    if len(trace_left) == 0 or len(trace_right) == 0:
        raise ValueError("One of the traces is empty.")

    n = min(len(trace_left), len(trace_right))

    if frame_indices is None:
        frame_indices = _default_frame_indices(trace_left[:n], max_frames=6)

    frame_indices = [i for i in frame_indices if 0 <= i < n]

    if len(frame_indices) == 0:
        raise ValueError("No valid frame indices were provided.")

    n_rows = len(frame_indices)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, 4.8 * n_rows),
        constrained_layout=True,
    )

    axes = np.array(axes).reshape(n_rows, 2)

    for row_idx, frame_idx in enumerate(frame_indices):
        state_left = trace_left[frame_idx]
        state_right = trace_right[frame_idx]

        _plot_single_publication_frame(
            ax=axes[row_idx, 0],
            state=state_left,
            frame_number=frame_idx,
            policy_name=left_name,
            xlim=xlim,
            ylim=ylim,
            show_legend_labels=(row_idx == 0),
        )

        _plot_single_publication_frame(
            ax=axes[row_idx, 1],
            state=state_right,
            frame_number=frame_idx,
            policy_name=right_name,
            xlim=xlim,
            ylim=ylim,
            show_legend_labels=False,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        f"Policy comparison: {left_name} vs {right_name}",
        fontsize=16,
        fontweight="bold",
        y=1.04,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return save_path


# ============================================================
# Internal helper functions
# ============================================================

def _plot_single_publication_frame(
    ax,
    state,
    frame_number: int,
    policy_name: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    show_legend_labels: bool = False,
):
    t = state["t"]
    pI = np.array(state["interceptor_pos"])
    threats = state["active_threats"]
    chosen_id = state["chosen_target_id"]
    features = state["features"]

    # Boundary
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.4,
        color="black",
        label="Boundary" if show_legend_labels else None,
    )

    # Interceptor
    ax.scatter(
        pI[0],
        pI[1],
        s=95,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="Interceptor" if show_legend_labels else None,
    )

    # Threats
    if threats:
        xs = [th["pos"][0] for th in threats]
        ys = [th["pos"][1] for th in threats]

        ax.scatter(
            xs,
            ys,
            s=75,
            color="#d62728",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            label="Threats" if show_legend_labels else None,
        )

    # Chosen target
    chosen_pos = None
    for th in threats:
        if th["id"] == chosen_id:
            chosen_pos = np.array(th["pos"])
            break

    if chosen_pos is not None:
        ax.scatter(
            chosen_pos[0],
            chosen_pos[1],
            s=240,
            facecolors="none",
            edgecolors="#2ca02c",
            linewidth=2.5,
            zorder=5,
            label="Chosen target" if show_legend_labels else None,
        )

        ax.annotate(
            "",
            xy=(chosen_pos[0], chosen_pos[1]),
            xytext=(pI[0], pI[1]),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.6,
                color="#2ca02c",
                shrinkA=6,
                shrinkB=8,
            ),
            zorder=2,
        )

    # Information box
    min_ttb = features.get("min_ttb", np.inf)
    min_pos_slack = features.get("min_positive_slack", np.inf)
    n_active = features.get("N_active", len(threats))
    neg_slack = features.get("count_negative_slack", 0)

    info = (
        f"t = {t:.1f}\n"
        f"N active = {n_active}\n"
        f"min TTB = {_fmt_num(min_ttb)}\n"
        f"min positive slack = {_fmt_num(min_pos_slack)}\n"
        f"negative slack = {neg_slack}\n"
        f"intercepted = {state['intercepted_so_far']}\n"
        f"escaped = {state['escaped_so_far']}"
    )

    ax.text(
        0.03,
        0.97,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="gray",
            alpha=0.90,
        ),
    )

    ax.set_title(
        f"{policy_name} | Frame {frame_number} | t={t:.1f}",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid(alpha=0.25)


def _default_frame_indices(trace, max_frames: int = 8) -> List[int]:
    """
    Selects useful frame indices.
    Prefer frames with active threats.
    """

    non_empty = [
        i for i, state in enumerate(trace)
        if len(state["active_threats"]) > 0
    ]

    if len(non_empty) >= max_frames:
        return np.linspace(
            non_empty[0],
            non_empty[-1],
            max_frames,
        ).astype(int).tolist()

    if len(non_empty) > 0:
        return non_empty

    return np.linspace(
        0,
        len(trace) - 1,
        min(max_frames, len(trace)),
    ).astype(int).tolist()


def _fmt_num(x) -> str:
    try:
        if np.isinf(x):
            return "∞"
        return f"{float(x):.2f}"
    except Exception:
        return str(x)
