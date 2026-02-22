from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Optional
import pandas as pd
from .env import ScenarioParams, SimEnv
from .heuristics import make_heuristics

def run_episode(params: ScenarioParams, heuristic_name: str, preempt: bool = False) -> Dict:
    env = SimEnv(params)
    heuristics = make_heuristics(seed=params.seed)
    h = heuristics[heuristic_name]

    target_id: Optional[int] = None

    while not env.done():
        active = env.active_threats()

        # decision points:
        # - if no target or target disappeared -> pick
        if target_id is None or all(th.id != target_id for th in active):
            target_id = h(active, env.interceptor_pos, params.v_interceptor)

        # step
        events = env.step(target_id)

        # event-driven preemption: if allowed and arrival occurred -> re-evaluate
        if preempt and events["arrival"] > 0:
            active2 = env.active_threats()
            target_id = h(active2, env.interceptor_pos, params.v_interceptor)

        # after intercept/escape, force re-eval next loop by clearing invalid target
        if events["intercept"] > 0 or events["escape"] > 0:
            # target might be gone; next iteration will fix it
            pass

    return {
        **asdict(params),
        "heuristic": heuristic_name,
        "preempt": preempt,
        "spawned": env.spawned,
        "intercepted": env.intercepted,
        "escaped": env.escaped,
    }

def compare_heuristics(params: ScenarioParams) -> pd.DataFrame:
    heuristics = list(make_heuristics(seed=params.seed).keys())
    rows = []
    for name in heuristics:
        rows.append(run_episode(params, name, preempt=False))
        # optional: preemption variants for NI and MPS
        if name in ("NI", "MPS"):
            rows.append(run_episode(params, name, preempt=True))
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["intercepted", "escaped"], ascending=[False, True]).reset_index(drop=True)
    return df
