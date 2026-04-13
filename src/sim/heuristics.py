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


def h_weighted(
    active: List[Threat],
    pI: np.ndarray,
    vI: float,
    w_tti: float = 1.0,
    w_ttb: float = 0.8,
) -> Optional[int]:
    if not active:
        return None

    def score(th: Threat) -> float:
        tti = time_to_intercept(pI, th.pos, vI)
        ttb = time_to_boundary_x0(th.pos, th.vel)
        return w_tti * tti + w_ttb * ttb

    best = min(active, key=score)
    return best.id


def h_cluster_first(
    active: List[Threat],
    pI: np.ndarray,
    vI: float,
    r: float = 15.0,
) -> Optional[int]:
    if not active:
        return None

    def neigh_count(th: Threat) -> int:
        return sum(
            1
            for other in active
            if other.id != th.id and np.linalg.norm(other.pos - th.pos) <= r
        )

    best = max(active, key=neigh_count)
    return best.id


def h_feasible_min_ttb(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    feasible = [
        (time_to_boundary_x0(th.pos, th.vel), th)
        for th in active
        if slack(pI, th, vI) >= 0
    ]
    if feasible:
        return min(feasible, key=lambda x: x[0])[1].id
    if not active:
        return None
    return min(active, key=lambda th: time_to_boundary_x0(th.pos, th.vel)).id


def h_max_speed(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    if not active:
        return None
    best = max(active, key=lambda th: float(np.linalg.norm(th.vel)))
    return best.id


def h_ratio_tti_ttb(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    if not active:
        return None

    def ratio(th: Threat) -> float:
        tti = time_to_intercept(pI, th.pos, vI)
        ttb = time_to_boundary_x0(th.pos, th.vel)
        return tti / max(ttb, 1e-6)

    best = min(active, key=ratio)
    return best.id


def h_danger_score(
    active: List[Threat],
    pI: np.ndarray,
    vI: float,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> Optional[int]:
    if not active:
        return None

    def danger(th: Threat) -> float:
        ttb = max(time_to_boundary_x0(th.pos, th.vel), 1e-6)
        tti = time_to_intercept(pI, th.pos, vI)
        return alpha / (ttb ** 2) + beta * tti

    best = max(active, key=danger)
    return best.id


def h_weighted_urgent(
    active: List[Threat],
    pI: np.ndarray,
    vI: float,
    w_tti: float = 0.5,
    w_ttb: float = 1.5,
) -> Optional[int]:
    return h_weighted(active, pI, vI, w_tti=w_tti, w_ttb=w_ttb)


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
        "FMTTB": h_feasible_min_ttb,
        "FastFirst": h_max_speed,
        "Ratio": h_ratio_tti_ttb,
        "Danger": h_danger_score,
        "WeightedU": lambda a, p, v: h_weighted_urgent(a, p, v, 0.5, 1.5),
    }
