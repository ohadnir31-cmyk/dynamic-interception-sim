from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Optional, List, Any
import pandas as pd

from .env import ScenarioParams, SimEnv, slack, time_to_boundary_x0, time_to_intercept
from .heuristics import make_heuristics


def _snapshot_features(env: SimEnv) -> Dict[str, Any]:
    active = env.active_threats()

    if not active:
        return {
            "N_active": 0,
            "min_ttb": float("inf"),
            "min_positive_slack": float("inf"),
            "count_negative_slack": 0,
        }

    ttbs = [time_to_boundary_x0(th.pos, th.vel) for th in active]
    slacks = [slack(env.interceptor_pos, th, env.p.v_interceptor) for th in active]
    pos_slacks = [s for s in slacks if s >= 0]

    return {
        "N_active": len(active),
        "min_ttb": float(min(ttbs)),
        "min_positive_slack": float(min(pos_slacks)) if pos_slacks else float("inf"),
        "count_negative_slack": int(sum(1 for s in slacks if s < 0)),
    }


def run_episode(params: ScenarioParams, heuristic_name: str, preempt: bool = False) -> Dict:
    env = SimEnv(params)
    heuristics = make_heuristics(seed=params.seed)
    h = heuristics[heuristic_name]

    target_id: Optional[int] = None

    while not env.done():
        active = env.active_threats()

        # decision point: if no target / target disappeared -> pick
        if target_id is None or all(th.id != target_id for th in active):
            target_id = h(active, env.interceptor_pos, params.v_interceptor)

        # advance one step
        events = env.step(target_id)

        # event-driven preemption: if allowed and a new target arrived -> re-evaluate
        if preempt and events["arrival"] > 0:
            active2 = env.active_threats()
            target_id = h(active2, env.interceptor_pos, params.v_interceptor)

    return {
        **asdict(params),
        "heuristic": heuristic_name,
        "preempt": preempt,
        "spawned": env.spawned,
        "intercepted": env.intercepted,
        "escaped": env.escaped,
    }


def run_episode_with_trace(
    params: ScenarioParams,
    heuristic_name: str,
    preempt: bool = False,
) -> Dict[str, Any]:
    env = SimEnv(params)
    heuristics = make_heuristics(seed=params.seed)
    h = heuristics[heuristic_name]

    target_id: Optional[int] = None
    trace: List[Dict[str, Any]] = []

    while not env.done():
        active_before = env.active_threats()

        # decision point
        if target_id is None or all(th.id != target_id for th in active_before):
            target_id = h(active_before, env.interceptor_pos, params.v_interceptor)

        feature_snapshot = _snapshot_features(env)

        trace.append({
            "t": float(env.t),
            "interceptor_pos": env.interceptor_pos.copy(),
            "chosen_target_id": target_id,
            "active_threats": [
                {
                    "id": th.id,
                    "pos": th.pos.copy(),
                    "vel": th.vel.copy(),
                    "ttb": float(time_to_boundary_x0(th.pos, th.vel)),
                    "tti": float(time_to_intercept(env.interceptor_pos, th.pos, params.v_interceptor)),
                    "slack": float(slack(env.interceptor_pos, th, params.v_interceptor)),
                }
                for th in active_before
            ],
            "features": feature_snapshot,
            "intercepted_so_far": int(env.intercepted),
            "escaped_so_far": int(env.escaped),
            "spawned_so_far": int(env.spawned),
        })

        events = env.step(target_id)

        # event-driven preemption
        if preempt and events["arrival"] > 0:
            active_after = env.active_threats()
            target_id = h(active_after, env.interceptor_pos, params.v_interceptor)

    summary = {
        **asdict(params),
        "heuristic": heuristic_name,
        "preempt": preempt,
        "spawned": env.spawned,
        "intercepted": env.intercepted,
        "escaped": env.escaped,
    }

    return {
        "summary": summary,
        "trace": trace,
    }


def compare_heuristics(params: ScenarioParams) -> pd.DataFrame:
    heuristics = list(make_heuristics(seed=params.seed).keys())
    rows = []

    for name in heuristics:
        rows.append(run_episode(params, name, preempt=False))

        # optional preemption variants
        if name in ("NI", "MPS"):
            rows.append(run_episode(params, name, preempt=True))

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["intercepted", "escaped"], ascending=[False, True]).reset_index(drop=True)
    return df
