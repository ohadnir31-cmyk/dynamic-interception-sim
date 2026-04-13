from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_trace(trace, boundary_x: float = 0.0, xlim=(-5, 80), ylim=(-60, 60), interval=300):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axvline(boundary_x, color="black", linestyle="--", linewidth=1.5, label="Boundary")

    interceptor_plot, = ax.plot([], [], "bo", markersize=8, label="Interceptor")
    threats_plot, = ax.plot([], [], "ro", linestyle="None", markersize=6, label="Threats")
    chosen_plot, = ax.plot([], [], "go", linestyle="None", markersize=10, label="Chosen target")

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
            f"min_TTB={f['min_ttb']:.2f}\n"
            f"min_pos_slack={f['min_positive_slack']:.2f}\n"
            f"neg_slack={f['count_negative_slack']}\n"
            f"intercepted={state['intercepted_so_far']}\n"
            f"escaped={state['escaped_so_far']}"
        )

        return interceptor_plot, threats_plot, chosen_plot, title, stats

    ani = FuncAnimation(fig, update, frames=len(trace), interval=interval, blit=False)
    plt.close(fig)
    return ani


def plot_static_paths(trace, boundary_x: float = 0.0, xlim=(-5, 80), ylim=(-60, 60)):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axvline(boundary_x, color="black", linestyle="--", linewidth=1.5, label="Boundary")

    interceptor_path = np.array([state["interceptor_pos"] for state in trace])
    ax.plot(interceptor_path[:, 0], interceptor_path[:, 1], "b-", linewidth=2, label="Interceptor path")

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
        ax.scatter(spawn_x, spawn_y, c="red", s=40, label="First seen positions")

    if last_seen:
        end_x = [p[0] for p in last_seen.values()]
        end_y = [p[1] for p in last_seen.values()]
        ax.scatter(end_x, end_y, c="gray", s=20, label="Last seen positions")

    ax.legend(loc="upper right")
    ax.set_title("Static path view")
    return fig, ax
