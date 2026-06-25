from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback if tqdm is unavailable
    tqdm = None

from src.sim.env import (
    SimEnv,
    ScenarioParams,
    slack,
    time_to_boundary_x0,
    time_to_intercept,
)
from src.sim.heuristics import make_heuristics


def n_active_bucket(n_active: int) -> str:
    if n_active <= 1:
        return "1"
    if n_active <= 3:
        return "2-3"
    if n_active <= 6:
        return "4-6"
    if n_active <= 10:
        return "7-10"
    return "11+"


def _safe_stat(values: np.ndarray, fn: str) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.inf

    if fn == "min":
        return float(np.min(finite))
    if fn == "mean":
        return float(np.mean(finite))
    if fn == "std":
        return float(np.std(finite))
    if fn == "median":
        return float(np.median(finite))

    raise ValueError(f"Unknown stat: {fn}")


def _remaining_horizon(env: SimEnv) -> float:
    return max(0.0, float(env.p.horizon_T) - float(env.t))


def _bounded_time_values(values: np.ndarray, cap: float) -> np.ndarray:
    """Return finite, finite-horizon time values for feature extraction.

    The simulator is finite-horizon. Very large TTB values caused by targets
    that barely move toward the boundary are not operationally distinct from
    any value larger than the remaining horizon. This helper maps inf and very
    large finite values to the remaining-horizon cap for state-feature
    statistics.
    """
    cap = max(0.0, float(cap))
    arr = np.array(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, cap)
    return np.minimum(arr, cap)


def _mean_nearest_neighbor_distance(positions: np.ndarray) -> float:
    if len(positions) <= 1:
        return np.inf

    distances = []
    for i in range(len(positions)):
        d = np.linalg.norm(positions[i] - positions, axis=1)
        d[i] = np.inf
        distances.append(np.min(d))

    return float(np.mean(distances))


def _progress(iterable, *, total: Optional[int], desc: str, disable: bool = False):
    """
    Small tqdm wrapper.

    This keeps the code usable even if tqdm is not installed, while showing a
    progress bar in Colab/Jupyter when tqdm is available.
    """
    if disable or tqdm is None:
        return iterable

    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def _validate_requested_heuristics(
    requested: Sequence[str],
    available: Dict[str, Any],
    label: str,
) -> None:
    missing = [h for h in requested if h not in available]
    if missing:
        raise ValueError(
            f"Unknown {label}: {missing}. "
            f"Available heuristics are: {list(available.keys())}"
        )


def extract_state_features(env: SimEnv) -> Dict[str, Any]:
    active = env.active_threats()

    if not active:
        return {
            "t": float(env.t),
            "interceptor_x": float(env.interceptor_pos[0]),
            "interceptor_y": float(env.interceptor_pos[1]),
            "N_active": 0,
            "N_active_bucket": "0",
            "min_ttb": np.inf,
            "mean_ttb": np.inf,
            "std_ttb": np.inf,
            "min_tti": np.inf,
            "mean_tti": np.inf,
            "std_tti": np.inf,
            "min_slack": np.inf,
            "mean_slack": np.inf,
            "std_slack": np.inf,
            "min_positive_slack": np.inf,
            "count_feasible": 0,
            "count_negative_slack": 0,
            "feasible_ratio": 0.0,
            "remaining_horizon": max(0.0, float(env.p.horizon_T) - float(env.t)),
            "ttb_capped_count": 0,
            "ttb_capped_ratio": 0.0,
            "cluster_index": np.inf,
            "spatial_spread_x": 0.0,
            "spatial_spread_y": 0.0,
            "spatial_dispersion": 0.0,
        }

    raw_ttbs = np.array([time_to_boundary_x0(th.pos, th.vel) for th in active], dtype=float)
    ttis = np.array(
        [time_to_intercept(env.interceptor_pos, th.pos, env.p.v_interceptor, th.vel) for th in active],
        dtype=float,
    )

    remaining_horizon = _remaining_horizon(env)
    ttbs = _bounded_time_values(raw_ttbs, remaining_horizon)

    # Feature-level slack is bounded by the finite remaining horizon. This is
    # different from the raw single-target slack used inside the heuristic
    # definitions, and is intentionally designed for learning and diagnostics.
    slacks = ttbs - ttis

    feasible_mask = slacks >= 0
    positive_slacks = slacks[feasible_mask]
    ttb_capped_mask = (~np.isfinite(raw_ttbs)) | (raw_ttbs > remaining_horizon)

    positions = np.array([th.pos for th in active], dtype=float)
    centroid = np.mean(positions, axis=0)

    spatial_spread_x = float(np.std(positions[:, 0])) if len(positions) > 1 else 0.0
    spatial_spread_y = float(np.std(positions[:, 1])) if len(positions) > 1 else 0.0
    spatial_dispersion = (
        float(np.mean(np.linalg.norm(positions - centroid, axis=1)))
        if len(positions) > 1
        else 0.0
    )

    return {
        "t": float(env.t),
        "interceptor_x": float(env.interceptor_pos[0]),
        "interceptor_y": float(env.interceptor_pos[1]),
        "N_active": int(len(active)),
        "N_active_bucket": n_active_bucket(len(active)),
        "min_ttb": _safe_stat(ttbs, "min"),
        "mean_ttb": _safe_stat(ttbs, "mean"),
        "std_ttb": _safe_stat(ttbs, "std"),
        "min_tti": _safe_stat(ttis, "min"),
        "mean_tti": _safe_stat(ttis, "mean"),
        "std_tti": _safe_stat(ttis, "std"),
        "min_slack": _safe_stat(slacks, "min"),
        "mean_slack": _safe_stat(slacks, "mean"),
        "std_slack": _safe_stat(slacks, "std"),
        "min_positive_slack": (
            float(np.min(positive_slacks)) if len(positive_slacks) > 0 else np.inf
        ),
        "count_feasible": int(np.sum(feasible_mask)),
        "count_negative_slack": int(np.sum(slacks < 0)),
        "feasible_ratio": float(np.mean(feasible_mask)),
        "remaining_horizon": float(remaining_horizon),
        "ttb_capped_count": int(np.sum(ttb_capped_mask)),
        "ttb_capped_ratio": float(np.mean(ttb_capped_mask)),
        "cluster_index": _mean_nearest_neighbor_distance(positions),
        "spatial_spread_x": spatial_spread_x,
        "spatial_spread_y": spatial_spread_y,
        "spatial_dispersion": spatial_dispersion,
    }


def rollout_from_env(
    env_snapshot: SimEnv,
    heuristic_name: str,
    preempt: bool = False,
) -> Dict[str, Any]:
    """
    Continue simulation from a copied environment using one fixed heuristic rule.
    """

    env = copy.deepcopy(env_snapshot)
    heuristics = make_heuristics(seed=env.p.seed)

    if heuristic_name not in heuristics:
        raise KeyError(
            f"Unknown rollout heuristic: {heuristic_name}. "
            f"Available: {list(heuristics.keys())}"
        )

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
    Evaluate all candidate heuristics from one state and determine the winner.

    Winner rule:
    1. maximize future_intercepted
    2. if tied, minimize future_escaped
    3. if still tied, keep a winner_set
    """

    results = []

    for h in candidate_heuristics:
        results.append(
            rollout_from_env(
                env_snapshot=env_snapshot,
                heuristic_name=h,
                preempt=preempt,
            )
        )

    best_intercepted = max(r["future_intercepted"] for r in results)
    candidates = [r for r in results if r["future_intercepted"] == best_intercepted]

    best_escaped = min(r["future_escaped"] for r in candidates)
    winners = [
        r["rollout_heuristic"]
        for r in candidates
        if r["future_escaped"] == best_escaped
    ]

    return {
        "winner": winners[0] if len(winners) == 1 else "TIE",
        "winner_set": ",".join(winners),
        "n_winners": len(winners),
        "best_future_intercepted": best_intercepted,
        "best_future_escaped": best_escaped,
        "rollout_details": results,
    }


def _rank_rollout_details(details: List[Dict[str, Any]]) -> Dict[str, int]:
    ranked_details = sorted(
        details,
        key=lambda r: (
            -r["future_intercepted"],
            r["future_escaped"],
            r["rollout_heuristic"],
        ),
    )

    return {
        r["rollout_heuristic"]: rank
        for rank, r in enumerate(ranked_details, start=1)
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
    Generate state-level rollout labels from one scenario.

    1. Run a behavior policy to generate realistic decision states.
    2. Copy the environment at selected states.
    3. Roll out each candidate heuristic from that state.
    4. Label the state using the best rollout result.
    """

    heuristics = make_heuristics(seed=params.seed)
    _validate_requested_heuristics([behavior_heuristic], heuristics, "behavior heuristic")
    _validate_requested_heuristics(candidate_heuristics, heuristics, "candidate heuristics")

    env = SimEnv(params)
    behavior_h = heuristics[behavior_heuristic]

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
                "scenario_regime": params.scenario_regime,
                "spatial_structure": params.spatial_structure,
                "arrival_process": params.arrival_process,
                "deadline_pressure": params.deadline_pressure,
                "initial_targets": params.initial_targets,
                "lambda_arrival": params.lambda_arrival,
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

            details = list(label["rollout_details"])
            rank_by_h = _rank_rollout_details(details)

            for r in details:
                h = r["rollout_heuristic"]
                row[f"{h}_future_intercepted"] = r["future_intercepted"]
                row[f"{h}_future_escaped"] = r["future_escaped"]
                row[f"{h}_regret"] = label["best_future_intercepted"] - r["future_intercepted"]
                row[f"{h}_rank"] = rank_by_h[h]

            rows.append(row)
            state_counter += 1

            if max_states is not None and state_counter >= max_states:
                break

        active = env.active_threats()

        if target_id is None or all(th.id != target_id for th in active):
            target_id = behavior_h(active, env.interceptor_pos, params.v_interceptor)

        events = env.step(target_id)

        if behavior_preempt and events["arrival"] > 0:
            active_after = env.active_threats()
            target_id = behavior_h(active_after, env.interceptor_pos, params.v_interceptor)

    return pd.DataFrame(rows)


def _scenario_behavior_jobs(
    scenarios: Dict[str, Any],
    behavior_heuristics: Sequence[str],
) -> List[Tuple[str, Any, str]]:
    jobs: List[Tuple[str, Any, str]] = []

    for scenario_name, scenario_obj in scenarios.items():
        for behavior_h in behavior_heuristics:
            jobs.append((scenario_name, scenario_obj, behavior_h))

    return jobs


def generate_dataset_for_scenarios(
    scenarios: Dict[str, Any],
    behavior_heuristics: Sequence[str],
    candidate_heuristics: Sequence[str],
    rollout_preempt: bool = False,
    max_states_per_run: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Generate rollout-labeled states for many scenarios and behavior heuristics.

    This is the expensive state-level counterfactual labeling stage. For each
    scenario and each behavior heuristic, up to max_states_per_run decision
    states are sampled. From each sampled state, all candidate heuristics are
    rolled out counterfactually.

    In Colab/Jupyter, show_progress=True displays a tqdm progress bar over
    scenario-behavior jobs. This makes long runs look alive instead of stuck.
    """

    if not scenarios:
        return pd.DataFrame()

    available = make_heuristics(seed=0)
    _validate_requested_heuristics(behavior_heuristics, available, "behavior heuristics")
    _validate_requested_heuristics(candidate_heuristics, available, "candidate heuristics")

    jobs = _scenario_behavior_jobs(scenarios, behavior_heuristics)
    total_jobs = len(jobs)

    print(
        "State-labeling jobs: "
        f"{len(scenarios)} scenarios × {len(behavior_heuristics)} behavior heuristics "
        f"= {total_jobs} jobs"
    )

    if max_states_per_run is not None:
        expected_counterfactual_rollouts = (
            len(scenarios)
            * len(behavior_heuristics)
            * max_states_per_run
            * len(candidate_heuristics)
        )
        print(
            "Upper-bound counterfactual rollouts: "
            f"{expected_counterfactual_rollouts:,} "
            f"({len(scenarios)} × {len(behavior_heuristics)} × "
            f"{max_states_per_run} × {len(candidate_heuristics)})"
        )

    all_dfs: List[pd.DataFrame] = []
    total_rows = 0

    iterator = _progress(
        jobs,
        total=total_jobs,
        desc="State-level labels",
        disable=not show_progress,
    )

    for job_index, (scenario_name, scenario_obj, behavior_h) in enumerate(iterator, start=1):
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

        if not df_part.empty:
            all_dfs.append(df_part)
            total_rows += len(df_part)

        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "scenario": scenario_name[:24],
                    "behavior": behavior_h,
                    "rows": total_rows,
                }
            )
        elif job_index % 50 == 0 or job_index == total_jobs:
            print(
                f"Completed {job_index}/{total_jobs} state-labeling jobs; "
                f"rows so far: {total_rows}"
            )

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)
