from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .env import Threat, time_to_boundary_x0, time_to_intercept, slack


HeuristicFn = Callable[[List[Threat], np.ndarray, float], Optional[int]]

EPS = 1e-9
DEFAULT_CLUSTER_TIME_WINDOW = 5.0


# ============================================================
# Final heuristic portfolio for the research proposal
# ============================================================
#
# Main portfolio:
#   NT       - nearest target by geometric distance
#   FNI      - nearest feasible target by moving-target lead-intercept time
#   FMTTB    - feasible target with minimum time-to-boundary
#   MPS      - feasible target with minimum positive slack
#   FCluster - leading edge of the densest local target cluster
#
# Notes:
#   - Danger and Ratio were intentionally removed from the main portfolio.
#   - All targets currently have equal value/priority.
#   - FCluster uses a mobility-based cluster radius:
#         r = vI * cluster_time_window
# ============================================================


def h_nearest(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    """
    NT - Nearest Target.

    Selects the active target that is geographically closest to the interceptor.
    This is the simplest proximity-based baseline.

    Does not explicitly consider feasibility, boundary urgency, or slack.
    """
    if not active:
        return None

    return min(active, key=lambda th: np.linalg.norm(th.pos - pI)).id


def h_feasible_nearest(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    """
    FNI - Feasible Nearest Intercept.

    First filters targets that are feasible to intercept before boundary crossing:
        slack = TTB - TTI >= 0

    Then selects the feasible target with the shortest moving-target lead-intercept time.

    Operational principle:
        feasibility-aware interceptability.
    """
    feasible = [th for th in active if slack(pI, th, vI) >= 0]

    if not feasible:
        return None

    return min(feasible, key=lambda th: time_to_intercept(pI, th.pos, vI, th.vel)).id


def h_feasible_min_ttb(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    """
    FMTTB - Feasible Minimum Time To Boundary.

    First filters targets that are feasible to intercept before boundary crossing:
        slack = TTB - TTI >= 0

    Then selects the feasible target with the shortest time-to-boundary.

    Operational principle:
        urgency-first under feasibility constraints.
    """
    feasible = [th for th in active if slack(pI, th, vI) >= 0]

    if not feasible:
        return None

    return min(feasible, key=lambda th: time_to_boundary_x0(th.pos, th.vel)).id


def h_min_positive_slack(active: List[Threat], pI: np.ndarray, vI: float) -> Optional[int]:
    """
    MPS - Minimum Positive Slack.

    Selects the feasible target with the smallest non-negative slack:
        slack = TTB - TTI

    This represents the most critical feasible interception opportunity:
    the target that can still be intercepted, but with the smallest remaining
    time margin.

    Operational principle:
        protect the most fragile remaining feasible opportunity.
    """
    feasible_with_slack = []

    for th in active:
        s = slack(pI, th, vI)
        if s >= 0:
            feasible_with_slack.append((s, th))

    if not feasible_with_slack:
        return None

    return min(feasible_with_slack, key=lambda item: item[0])[1].id


def h_frontier_cluster(
    active: List[Threat],
    pI: np.ndarray,
    vI: float,
    cluster_time_window: float = DEFAULT_CLUSTER_TIME_WINDOW,
) -> Optional[int]:
    """
    FCluster - Frontier-Cluster.

    Defines a local cluster radius according to interceptor mobility:
        r = vI * cluster_time_window

    For each active target, the heuristic defines its local neighborhood as all
    active targets within distance r. It then selects the densest local
    neighborhood. From that neighborhood, it selects the geographically leading
    target, i.e., the target with the smallest x-coordinate, assuming the
    protected boundary is x = 0 and targets approach it from x > 0.

    Tie-breaking:
        1. Prefer the densest local neighborhood.
        2. If there is a tie, prefer the neighborhood whose leading target
           is closest to the protected boundary.
        3. If there is still a tie, prefer the neighborhood whose seed target
           is closer to the interceptor.

    Operational principle:
        spatial positioning at the leading edge of a dense target group.
        The intuition is that after moving toward the front of a dense group,
        additional targets from the same group may pass near the interceptor,
        requiring only limited lateral movement for future interceptions.
    """
    if not active:
        return None

    r = max(float(vI) * float(cluster_time_window), EPS)

    def cluster_around(seed: Threat) -> List[Threat]:
        return [
            other
            for other in active
            if np.linalg.norm(other.pos - seed.pos) <= r
        ]

    clusters = [(seed, cluster_around(seed)) for seed in active]

    seed, chosen_cluster = max(
        clusters,
        key=lambda item: (
            len(item[1]),
            -min(member.pos[0] for member in item[1]),
            -np.linalg.norm(item[0].pos - pI),
        ),
    )

    return min(chosen_cluster, key=lambda th: th.pos[0]).id


# ============================================================
# Registry
# ============================================================

def make_heuristics(seed: int = 0) -> Dict[str, HeuristicFn]:
    """
    Return the final heuristic portfolio used by the current research proposal.

    The seed argument is kept for compatibility with existing experiment code.
    The current portfolio is deterministic and does not use the seed.
    """

    return {
        # NT is the current name used in proposal figures and future outputs.
        # NI is kept as a backwards-compatible alias for datasets generated before the rename.
        "NT": h_nearest,
        "NI": h_nearest,
        "FNI": h_feasible_nearest,
        "FMTTB": h_feasible_min_ttb,
        "MPS": h_min_positive_slack,
        "FCluster": h_frontier_cluster,
    }
