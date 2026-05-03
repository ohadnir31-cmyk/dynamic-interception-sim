from __future__ import annotations

from typing import Callable, Dict, List, Optional
import numpy as np

from .env import Threat, time_to_boundary_x0, time_to_intercept, slack


HeuristicFn = Callable[[List[Threat], np.ndarray, float], Optional[int]]


EPS = 1e-6


# ============================================================
# Basic heuristics
# ============================================================

def h_nearest(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    if not active:
        return None
    return min(active, key=lambda th: np.linalg.norm(th.pos - pI)).id


def h_min_ttb(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    if not active:
        return None
    return min(active, key=lambda th: time_to_boundary_x0(th.pos, th.vel)).id


def h_min_positive_slack(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    feasible = [(slack(pI, th, vI), th) for th in active if slack(pI, th, vI) >= 0]
    if not feasible:
        return None
    return min(feasible, key=lambda x: x[0])[1].id


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

    return min(active, key=score).id


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

    return max(active, key=neigh_count).id


def h_random_factory(seed: int = 0) -> HeuristicFn:
    rng = np.random.default_rng(seed)

    def h_random(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
        if not active:
            return None
        return int(rng.choice([th.id for th in active]))

    return h_random


# ============================================================
# Stronger / research-useful heuristics
# ============================================================

def h_feasible_nearest(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    """
    Nearest target among feasible targets only.
    If no feasible target exists, return None.
    """
    feasible = [th for th in active if slack(pI, th, vI) >= 0]
    if not feasible:
        return None
    return min(feasible, key=lambda th: time_to_intercept(pI, th.pos, vI)).id


def h_feasible_min_ttb(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    """
    Earliest boundary crossing among feasible targets only.
    If no feasible target exists, return None.
    """
    feasible = [th for th in active if slack(pI, th, vI) >= 0]
    if not feasible:
        return None
    return min(feasible, key=lambda th: time_to_boundary_x0(th.pos, th.vel)).id


def h_min_ratio_tti_ttb(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    """
    Select target with minimal TTI / TTB.
    Captures relative reachability rather than absolute slack.
    """
    if not active:
        return None

    def score(th: Threat) -> float:
        tti = time_to_intercept(pI, th.pos, vI)
        ttb = time_to_boundary_x0(th.pos, th.vel)
        return tti / max(ttb, EPS)

    return min(active, key=score).id


def h_max_margin(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    """
    Select feasible target with largest positive slack.
    This is intentionally different from MPS:
    MPS saves the most urgent feasible target.
    MaxMargin prefers the safest feasible target, often preserving throughput.
    """
    feasible = [(slack(pI, th, vI), th) for th in active if slack(pI, th, vI) >= 0]
    if not feasible:
        return None
    return max(feasible, key=lambda x: x[0])[1].id


def h_danger_score(
    active: List[Threat],
    pI: np.ndarray,
    vI: float,
    alpha: float = 1.5,
    beta: float = 1.0,
    gamma: float = 2.0,
) -> Optional[int]:
    """
    Composite danger heuristic.

    High score means:
    - small TTB = urgent
    - small TTI = reachable
    - negative slack penalty discourages impossible targets
    """
    if not active:
        return None

    def score(th: Threat) -> float:
        ttb = max(time_to_boundary_x0(th.pos, th.vel), EPS)
        tti = max(time_to_intercept(pI, th.pos, vI), EPS)
        s = slack(pI, th, vI)

        infeasible_penalty = gamma * max(-s, 0.0)
        return alpha / ttb + beta / tti - infeasible_penalty

    return max(active, key=score).id


def h_density_urgent(
    active: List[Threat],
    pI: np.ndarray,
    vI: float,
    r: float = 15.0,
    w_density: float = 1.0,
    w_ttb: float = 1.0,
    w_tti: float = 0.5,
) -> Optional[int]:
    """
    Combines local density, urgency, and reachability.

    Useful when clustered threats exist but not all clustered targets are equally urgent.
    """
    if not active:
        return None

    def density(th: Threat) -> int:
        return sum(
            1
            for other in active
            if other.id != th.id and np.linalg.norm(other.pos - th.pos) <= r
        )

    def score(th: Threat) -> float:
        d = density(th)
        ttb = max(time_to_boundary_x0(th.pos, th.vel), EPS)
        tti = max(time_to_intercept(pI, th.pos, vI), EPS)

        return w_density * d + w_ttb / ttb + w_tti / tti

    return max(active, key=score).id


def h_opportunity_cost(
    active: List[Threat],
    pI: np.ndarray,
    vI: float,
    r: float = 18.0,
) -> Optional[int]:
    """
    Select target that appears to preserve future opportunities.

    The score rewards targets that are:
    - feasible
    - near other targets
    - not too far from interceptor
    """
    feasible = [th for th in active if slack(pI, th, vI) >= 0]
    if not feasible:
        return None

    def local_future_value(th: Threat) -> float:
        neighbors = [
            other
            for other in active
            if other.id != th.id and np.linalg.norm(other.pos - th.pos) <= r
        ]

        neighbor_value = 0.0
        for other in neighbors:
            neighbor_ttb = max(time_to_boundary_x0(other.pos, other.vel), EPS)
            neighbor_value += 1.0 / neighbor_ttb

        tti = time_to_intercept(pI, th.pos, vI)
        s = slack(pI, th, vI)

        return neighbor_value + 0.5 * s - 0.3 * tti

    return max(feasible, key=local_future_value).id


def h_short_lookahead_greedy(
    active: List[Threat],
    pI: np.ndarray,
    vI: float,
) -> Optional[int]:
    """
    Lightweight one-step lookahead.

    Does not simulate the full environment.
    It estimates whether choosing a target leaves the interceptor
    near useful future targets.
    """
    if not active:
        return None

    def score(th: Threat) -> float:
        tti = time_to_intercept(pI, th.pos, vI)
        ttb = time_to_boundary_x0(th.pos, th.vel)
        s = ttb - tti

        if s < 0:
            return -1e6 + s

        # Estimated interceptor position after reaching this target
        p_after = th.pos

        future_reachable = 0
        future_urgency = 0.0

        for other in active:
            if other.id == th.id:
                continue

            other_tti = time_to_intercept(p_after, other.pos, vI)
            other_ttb = time_to_boundary_x0(other.pos, other.vel)

            if other_ttb - other_tti >= 0:
                future_reachable += 1
                future_urgency += 1.0 / max(other_ttb, EPS)

        return 2.0 * future_reachable + future_urgency - 0.2 * tti

    return max(active, key=score).id


# ============================================================
# Registry
# ============================================================

def make_heuristics(seed: int = 0) -> Dict[str, HeuristicFn]:
    """
    Return all heuristic policies.

    Keep names short because they become labels in datasets.
    """

    return {
        # Original / baseline
        "NI": h_nearest,
        "TTB": h_min_ttb,
        "MPS": h_min_positive_slack,
        "Weighted": lambda a, p, v: h_weighted(a, p, v, 1.0, 0.8),
        "Cluster": lambda a, p, v: h_cluster_first(a, p, v, r=15.0),
        "Random": h_random_factory(seed),

        # Stronger policies
        "FNI": h_feasible_nearest,
        "FMTTB": h_feasible_min_ttb,
        "Ratio": h_min_ratio_tti_ttb,
        "MaxMargin": h_max_margin,
        "Danger": h_danger_score,
        "DensityUrgent": h_density_urgent,
        "OppCost": h_opportunity_cost,
        "Lookahead": h_short_lookahead_greedy,
    }
