from __future__ import annotations
from typing import List, Optional, Callable, Dict
import numpy as np
from .env import Threat, time_to_boundary_x0, time_to_intercept, slack

HeuristicFn = Callable[[List[Threat], np.ndarray, float], Optional[int]]

def h_nearest(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    if not active:
        return None
    best = min(active, key=lambda th: np.linalg.norm(th.pos - pI))
    return best.id

def h_min_ttb(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    if not active:
        return None
    best = min(active, key=lambda th: time_to_boundary_x0(th.pos, th.vel))
    return best.id

def h_min_positive_slack(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    feasible = []
    for th in active:
        s = slack(pI, th, vI)
        if s >= 0:
            feasible.append((s, th))
    if not feasible:
        return None
    feasible.sort(key=lambda x: x[0])
    return feasible[0][1].id

def h_weighted(active: List[Threat], pI: np.ndarray, vI: float, w_tti: float = 1.0, w_ttb: float = 0.8) -> Optional[int]:
    if not active:
        return None
    def score(th: Threat) -> float:
        tti = time_to_intercept(pI, th.pos, vI)
        ttb = time_to_boundary_x0(th.pos, th.vel)
        return w_tti * tti + w_ttb * ttb
    best = min(active, key=score)
    return best.id

def h_cluster_first(active: List[Threat], pI: np.ndarray, vI: float, r: float = 15.0) -> Optional[int]:
    if not active:
        return None
    # choose threat with maximum neighbors within radius r
    def neigh_count(th: Threat) -> int:
        c = 0
        for other in active:
            if other.id == th.id:
                continue
            if np.linalg.norm(other.pos - th.pos) <= r:
                c += 1
        return c
    best = max(active, key=neigh_count)
    return best.id

def make_heuristics(seed: int = 0) -> Dict[str, HeuristicFn]:
    rng = np.random.default_rng(seed)

    def h_random(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
        if not active:
            return None
        return int(rng.choice([th.id for th in active]))

    return {
        "NI": h_nearest,
        "TTB": h_min_ttb,
        "MPS": h_min_positive_slack,
        "Weighted": lambda a, p, v: h_weighted(a, p, v, 1.0, 0.8),
        "Cluster": lambda a, p, v: h_cluster_first(a, p, v, r=15.0),
        "Random": h_random,
    }
