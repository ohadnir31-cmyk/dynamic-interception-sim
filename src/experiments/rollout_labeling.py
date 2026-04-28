from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Any

import numpy as np
import pandas as pd

from src.sim.env import (
    SimEnv,
    ScenarioParams,
    slack,
    time_to_boundary_x0,
    time_to_intercept,
)
from src.sim.heuristics import make_heuristics


def extract_state_features(env: SimEnv) -> Dict[str, Any]:
    active = env.active_threats()

    if not active:
        return {
            "t": env.t,
            "interceptor_x": float(env.interceptor_pos[0]),
            "interceptor_y": float(env.interceptor_pos[1]),
            "N_active": 0,
            "min_ttb": np.inf,
            "mean_ttb": np.inf,
            "min_positive_slack": np.inf,
            "mean_slack": np.inf,
            "count_negative_slack": 0,
            "mean_tti": np.inf,
            "cluster_index": np.inf,
        }

    ttbs = np.array([time_to_boundary_x0(th.pos, th.vel) for th in active], dtype=float)
    ttis = np.array([time_to_intercept(env.interceptor_pos, th.pos, env.p.v_interceptor) for th in active], dtype=float)
    slacks = np.array([slack(env.interceptor_pos, th, env.p.v_interceptor) for th in active], dtype=float)

    positive_slacks = slacks[slacks >= 0]

    positions = np.array([th.pos for th in active], dtype=float)
    cluster_index = _mean_nearest_neighbor_distance(positions)

    return {
        "t": float(env.t),
        "interceptor_x": float(env.interceptor_pos[0]),
        "interceptor_y": float(env.interceptor_pos[1]),
        "N_active": int(len(active)),
        "min_ttb": float(np.min(ttbs)),
        "mean_ttb": float(np.mean(ttbs)),
        "min_positive_slack": float(np.min(positive_slacks)) if len(positive_slacks) > 0 else np.inf,
        "mean_slack": float(np.mean(slacks)),
        "count_negative_slack": int(np.sum(slacks < 0)),
        "mean_tti": float(np.mean(ttis)),
        "cluster_index": float(cluster_index),
    }


def _mean_nearest_neighbor_distance(positions: np.ndarray) -> float:
    if len(positions) <= 1:
        return np.inf

    distances = []
    for i in range(len(positions)):
        d = np.linalg.norm(positions[i] - positions, axis=1)
        d[i] = np.inf
        distances.append(np.min(d))

    return float(np.mean(distances))


def rollout_from_env(
    env_snapshot: SimEnv,
    heuristic_name: str,
    preempt: bool = False,
) -> Dict[str, Any]:
    """
    Continue simulation from a copied environment using one fixed heuristic rule.

    Important:
    - The heuristic rule remains fixed.
    - The selected target may change when the current target disappears.
    - If preempt=True, the heuristic may also re-evaluate after new arrivals.
    """

    env = copy.deepcopy(env_snapshot)
    heuristics = make_heuristics(seed=env.p.seed)
    h = heuristics[heuristic_name]

    start_intercepted = env.intercepted
    start_escaped = env.escaped
    start_spawned = env.spawned

    target_id: Optional[int] = None

    while not env.done():
        active = env.active_threats()

        if target_id is None or all(th.id != target_id for th in active):
            target_id = h(active, env.interceptor_pos, env.p.v_interceptor)

        events = env.step(target_id)

        if preempt and events["arrival"] > 0:
            active_after = env.active_threats()
            target_id = h(active_after, env.interceptor_pos, env.p.v_interceptor)

    return {
        "rollout_heuristic": heuristic_name,
        "rollout_preempt": preempt,
        "future_intercepted": env.intercepted - start_intercepted,
        "future_escaped": env.escaped - start_escaped,
        "future_spawned": env.spawned - start_spawned,
        "final_intercepted": env.intercepted,
        "final_escaped": env.escaped,
        "final_spawned": env.spawned,
    }


def label_state_by_rollout(
    env_snapshot: SimEnv,
    candidate_heuristics: Sequence[str],
    preempt: bool = False,
) -> Dict[str, Any]:
    """
    For one state, evaluate each heuristic by rollout and choose the best one.

    Label rule:
    1. maximize future_intercepted
    2. tie-break by minimizing future_escaped
    3. if still tied, keep all winners in winner_set
    """

    results = []

    for h in candidate_heuristics:
        r = rollout_from_env(
            env_snapshot=env_snapshot,
            heuristic_name=h,
            preempt=preempt,
        )
        results.append(r)

    best_intercepted = max(r["future_intercepted"] for r in results)
    candidates = [r for r in results if r["future_intercepted"] == best_intercepted]

    best_escaped = min(r["future_escaped"] for r in candidates)
    winners = [r["rollout_heuristic"] for r in candidates if r["future_escaped"] == best_escaped]

    return {
        "winner": winners[0] if len(winners) == 1 else "TIE",
        "winner_set": ",".join(winners),
        "n_winners": len(winners),
        "best_future_intercepted": best_intercepted,
        "best_future_escaped": best_escaped,
        "rollout_details": results,
    }


def generate_rollout_labeled_dataset(
    params: ScenarioParams,
    scenario_name: str,
    behavior_heuristic: str,
    candidate_heuristics: Sequence[str],
    behavior_preempt: bool = False,
    rollout_preempt: bool = False,
    decision_only: bool = True,
    max_states: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate rollout labels from one scenario.

    Steps:
    1. Run a behavior policy to generate realistic states.
    2. At selected decision states, copy the environment.
    3. Roll out every candidate heuristic from that state.
    4. Label the state with the best rollout heuristic.

    behavior_heuristic:
        The heuristic used to generate states.

    candidate_heuristics:
        The heuristics evaluated as labels.

    decision_only:
        If True, label only states with at least one active target.
        If False, label every time step.
    """

    env = SimEnv(params)
    behavior_h = make_heuristics(seed=params.seed)[behavior_heuristic]

    rows: List[Dict[str, Any]] = []
    target_id: Optional[int] = None
    state_counter = 0

    while not env.done():
        active = env.active_threats()

        should_label = (len(active) > 0) if decision_only else True

        if should_label:
            env_snapshot = copy.deepcopy(env)

            features = extract_state_features(env_snapshot)
            label = label_state_by_rollout(
                env_snapshot=env_snapshot,
                candidate_heuristics=candidate_heuristics,
                preempt=rollout_preempt,
            )

            row = {
                "scenario": scenario_name,
                "seed": params.seed,
                "behavior_heuristic": behavior_heuristic,
                "behavior_preempt": behavior_preempt,
                "rollout_preempt": rollout_preempt,
                "state_id": state_counter,
                **features,
                "winner": label["winner"],
                "winner_set": label["winner_set"],
                "n_winners": label["n_winners"],
                "best_future_intercepted": label["best_future_intercepted"],
                "best_future_escaped": label["best_future_escaped"],
            }

            # Add per-heuristic rollout columns
            for r in label["rollout_details"]:
                h = r["rollout_heuristic"]
                row[f"{h}_future_intercepted"] = r["future_intercepted"]
                row[f"{h}_future_escaped"] = r["future_escaped"]

            rows.append(row)
            state_counter += 1

            if max_states is not None and state_counter >= max_states:
                break

        # Normal behavior-policy step
        active = env.active_threats()

        if target_id is None or all(th.id != target_id for th in active):
            target_id = behavior_h(active, env.interceptor_pos, params.v_interceptor)

        events = env.step(target_id)

        if behavior_preempt and events["arrival"] > 0:
            active_after = env.active_threats()
            target_id = behavior_h(active_after, env.interceptor_pos, params.v_interceptor)

    return pd.DataFrame(rows)


def generate_dataset_for_scenarios(
    scenarios: Dict[str, Any],
    behavior_heuristics: Sequence[str],
    candidate_heuristics: Sequence[str],
    rollout_preempt: bool = False,
    max_states_per_run: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate rollout-labeled data for multiple scenarios and behavior heuristics.

    Recommended initially:
    behavior_heuristics = ["NI", "MPS", "Cluster"]
    candidate_heuristics = ["NI", "TTB", "MPS", "Weighted", "Cluster"]
    """

    all_dfs = []

    for scenario_name, scenario_obj in scenarios.items():
        for behavior_h in behavior_heuristics:
            df_part = generate_rollout_labeled_dataset(
                params=scenario_obj.params,
                scenario_name=scenario_name,
                behavior_heuristic=behavior_h,
                candidate_heuristics=candidate_heuristics,
                behavior_preempt=False,
                rollout_preempt=rollout_preempt,
                decision_only=True,
                max_states=max_states_per_run,
            )

            all_dfs.append(df_part)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)
