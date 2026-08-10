from __future__ import annotations

"""Exact adaptive heuristic-portfolio oracle for deliberately small scenarios.

The search enumerates every distinct target action proposed by the heuristic
portfolio at each pursued-target decision epoch.  Heuristics that propose the
same target share one branch.  The resulting oracle is exact *within the
implemented heuristic portfolio*; it is not an unrestricted target-level
optimum.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.experiments.closed_loop_fc_selector import (
    DEFAULT_CANDIDATE_HEURISTICS,
    advance_one_pursuit,
    canonical_code_name,
    display_heuristic_name,
    proposed_targets_by_heuristic,
)
from src.sim.env import ScenarioParams, SimEnv


class OracleSearchLimitExceeded(RuntimeError):
    """Raised when a supposedly small exact-search case exceeds a safety limit."""


@dataclass
class OracleSearchStats:
    nodes: int = 0
    leaves: int = 0
    deduplicated_heuristic_proposals: int = 0
    maximum_depth_reached: int = 0


@dataclass
class OracleOutcome:
    intercepted: int
    escaped: int
    spawned: int
    active_at_horizon: int
    decisions: int
    path: List[Dict[str, Any]] = field(default_factory=list)
    stats: OracleSearchStats = field(default_factory=OracleSearchStats)


def _advance_to_decision_epoch(env: SimEnv) -> None:
    """Advance idle time until at least one target is active or the horizon ends."""
    while not env.done() and not env.active_threats():
        env.step(None)


def _outcome_key(outcome: OracleOutcome) -> tuple[Any, ...]:
    # Main objective: maximize interceptions.  Remaining criteria are only
    # deterministic tie-breakers and do not redefine the research objective.
    path_codes = tuple(str(item["representative_heuristic_code"]) for item in outcome.path)
    return (
        int(outcome.intercepted),
        -int(outcome.escaped),
        -int(outcome.active_at_horizon),
        tuple(reversed(path_codes)),
    )


def exact_adaptive_portfolio_oracle(
    params: ScenarioParams,
    *,
    heuristic_names: Sequence[str] = DEFAULT_CANDIDATE_HEURISTICS,
    max_decisions: int = 6,
    max_nodes: int = 250_000,
    require_valid_proposal_from_all_heuristics: bool = True,
) -> OracleOutcome:
    """Enumerate the exact adaptive portfolio optimum for one small scenario.

    Parameters
    ----------
    max_decisions:
        Hard safety limit.  If any branch still contains a decision after this
        depth, the result is not exact and an exception is raised.
    max_nodes:
        Hard cap protecting Colab from accidental combinatorial explosion.
    require_valid_proposal_from_all_heuristics:
        The default small-suite generator creates states in which all five
        heuristics have an actionable target.  Requiring this property avoids
        adding repeated one-step wait actions to the search tree and makes the
        exact oracle directly comparable to the intended adaptive selector.
    """

    codes = [canonical_code_name(name) for name in heuristic_names]
    stats = OracleSearchStats()
    root = SimEnv(params)

    def search(env: SimEnv, depth: int) -> OracleOutcome:
        _advance_to_decision_epoch(env)
        stats.nodes += 1
        stats.maximum_depth_reached = max(stats.maximum_depth_reached, depth)
        if stats.nodes > int(max_nodes):
            raise OracleSearchLimitExceeded(
                f"Exact search exceeded max_nodes={max_nodes:,}. "
                "Reduce targets/horizon or increase the explicit safety limit."
            )

        if env.done():
            stats.leaves += 1
            return OracleOutcome(
                intercepted=int(env.intercepted),
                escaped=int(env.escaped),
                spawned=int(env.spawned),
                active_at_horizon=int(
                    env.spawned - env.intercepted - env.escaped
                ),
                decisions=depth,
                path=[],
                stats=stats,
            )

        if depth >= int(max_decisions):
            raise OracleSearchLimitExceeded(
                f"A branch requires more than max_decisions={max_decisions}; "
                "the requested exact result would be truncated."
            )

        proposals = proposed_targets_by_heuristic(env, codes)
        invalid = [code for code, target in proposals.items() if target is None]
        if invalid and require_valid_proposal_from_all_heuristics:
            raise OracleSearchLimitExceeded(
                "The small exact-oracle scenario contains non-actionable "
                f"heuristics at t={env.t:.3f}: {invalid}. Use a more feasible "
                "small scenario or explicitly allow wait actions."
            )

        action_groups: Dict[int, List[str]] = {}
        for code in codes:
            target_id = proposals.get(code)
            if target_id is None:
                continue
            action_groups.setdefault(int(target_id), []).append(code)

        if not action_groups:
            raise OracleSearchLimitExceeded(
                "No heuristic proposed an actionable target at a nonterminal state."
            )

        stats.deduplicated_heuristic_proposals += len(codes) - len(action_groups)
        best: Optional[OracleOutcome] = None

        for target_id, equivalent_codes in action_groups.items():
            child = copy.deepcopy(env)
            representative = equivalent_codes[0]
            transition = advance_one_pursuit(
                child,
                representative,
                selected_target_id=int(target_id),
            )
            continuation = search(child, depth + 1)
            decision = {
                "decision_index": depth,
                "start_time": float(transition.start_time),
                "end_time": float(transition.end_time),
                "selected_target_id": int(target_id),
                "representative_heuristic_code": representative,
                "representative_heuristic": display_heuristic_name(representative),
                "equivalent_heuristic_codes": list(equivalent_codes),
                "equivalent_heuristics": [
                    display_heuristic_name(code) for code in equivalent_codes
                ],
                "termination_reason": transition.termination_reason,
                "interceptions_during_pursuit": int(transition.interceptions),
                "escapes_during_pursuit": int(transition.escapes),
                "arrivals_during_pursuit": int(transition.arrivals),
            }
            candidate = OracleOutcome(
                intercepted=continuation.intercepted,
                escaped=continuation.escaped,
                spawned=continuation.spawned,
                active_at_horizon=continuation.active_at_horizon,
                decisions=continuation.decisions,
                path=[decision, *continuation.path],
                stats=stats,
            )
            if best is None or _outcome_key(candidate) > _outcome_key(best):
                best = candidate

        if best is None:  # pragma: no cover - guarded above
            raise RuntimeError("Exact portfolio search produced no candidate branch.")
        return best

    outcome = search(root, 0)
    outcome.stats = stats
    return outcome


def make_small_oracle_scenarios(
    n_scenarios: int,
    *,
    seed: int = 20260811,
    min_targets: int = 3,
    max_targets: int = 5,
    horizon: float = 8.0,
    dt: float = 0.25,
    require_oracle_advantage: bool = True,
    max_candidate_scenarios: int = 5_000,
) -> Dict[str, ScenarioParams]:
    """Create deterministic nontrivial manual cases for exact enumeration.

    Candidates are accepted only when the fixed heuristics do not all obtain
    the same score.  By default the exact adaptive portfolio oracle must also
    exceed the best fixed heuristic by at least one interception.  This avoids
    filling the appendix with uninformative cases in which every policy captures
    every target.
    """

    if min_targets < 1 or max_targets < min_targets:
        raise ValueError("Invalid min_targets/max_targets range.")
    rng = np.random.default_rng(seed)
    from src.sim.runner import run_episode

    scenarios: Dict[str, ScenarioParams] = {}
    candidate_index = 0
    while (
        len(scenarios) < int(n_scenarios)
        and candidate_index < int(max_candidate_scenarios)
    ):
        index = candidate_index
        candidate_index += 1
        target_count = int(rng.integers(min_targets, max_targets + 1))
        candidate_horizon = float(
            rng.choice(
                sorted(
                    {
                        max(4.0, float(horizon) - 2.0),
                        max(5.0, float(horizon) - 1.0),
                        float(horizon),
                        float(horizon) + 1.0,
                    }
                )
            )
        )
        manual_threats: List[Dict[str, Any]] = []
        for target_index in range(target_count):
            # Most targets are available at t=0; an optional late target retains
            # a small amount of dynamic-arrival structure without increasing the
            # total number of decisions beyond max_targets.
            birth = 0.0
            if target_index == target_count - 1 and target_count >= 4:
                birth = float(rng.choice([0.0, 0.5, 1.0, 1.5, 2.0]))
            x = float(rng.uniform(0.8, 5.5))
            y = float(rng.uniform(-5.0, 5.0))
            vx = -float(rng.uniform(0.05, 0.65))
            vy = float(rng.uniform(-0.12, 0.12))
            manual_threats.append(
                {"t": birth, "pos": [x, y], "vel": [vx, vy]}
            )

        params = ScenarioParams(
            seed=seed * 10_000 + index,
            horizon_T=candidate_horizon,
            dt=float(dt),
            lambda_arrival=0.0,
            initial_targets=0,
            arrival_process="poisson",
            spatial_structure="uniform",
            scenario_regime="small_exact",
            deadline_pressure="moderate",
            v_interceptor=1.0,
            kill_radius=0.12,
            home=(0.0, 0.0),
            manual_threats=manual_threats,
        )

        fixed_scores = {
            code: int(run_episode(params, code, preempt=False)["intercepted"])
            for code in DEFAULT_CANDIDATE_HEURISTICS
        }
        if len(set(fixed_scores.values())) < 2:
            continue

        if require_oracle_advantage:
            try:
                oracle = exact_adaptive_portfolio_oracle(
                    params,
                    heuristic_names=DEFAULT_CANDIDATE_HEURISTICS,
                    max_decisions=max_targets + 1,
                    max_nodes=250_000,
                    require_valid_proposal_from_all_heuristics=True,
                )
            except OracleSearchLimitExceeded:
                continue
            if int(oracle.intercepted) <= max(fixed_scores.values()):
                continue

        accepted_index = len(scenarios)
        name = (
            f"small_exact_{accepted_index:03d}_candidate{index:04d}_"
            f"n{target_count}"
        )
        scenarios[name] = params

    if len(scenarios) < int(n_scenarios):
        raise OracleSearchLimitExceeded(
            f"Found only {len(scenarios)} suitable nontrivial small scenarios "
            f"after {max_candidate_scenarios} candidates. Reduce n_scenarios, "
            "relax require_oracle_advantage, or increase the candidate limit."
        )

    return scenarios
