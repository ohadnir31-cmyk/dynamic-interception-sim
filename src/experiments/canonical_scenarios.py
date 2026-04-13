from dataclasses import dataclass
from typing import Dict

from src.sim.env import ScenarioParams


@dataclass(frozen=True)
class CanonicalScenario:
    name: str
    description: str
    params: ScenarioParams


def get_canonical_scenarios() -> Dict[str, CanonicalScenario]:
    scenarios = {}

    # -------------------------------------------------
    # Scenario A: Low load, long deadlines
    # -------------------------------------------------
    scenarios["A_low_load_long_deadlines"] = CanonicalScenario(
        name="A_low_load_long_deadlines",
        description="Few threats, far from boundary, slow threats. Sanity-check regime.",
        params=ScenarioParams(
            seed=0,
            horizon_T=60.0,
            dt=0.5,
            lambda_arrival=0.05,
            x_spawn_mean=60.0,
            x_spawn_std=5.0,
            y_spawn_sigma=30.0,
            v_threat_mean=10.0,
            v_threat_std=1.0,
            v_interceptor=25.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
        ),
    )

    # -------------------------------------------------
    # Scenario B: Tight deadlines
    # -------------------------------------------------
    scenarios["B_tight_deadlines"] = CanonicalScenario(
        name="B_tight_deadlines",
        description="Threats appear close to boundary and move fast (deadline-dominated).",
        params=ScenarioParams(
            seed=0,
            horizon_T=60.0,
            dt=0.5,
            lambda_arrival=0.10,
            x_spawn_mean=20.0,
            x_spawn_std=3.0,
            y_spawn_sigma=30.0,
            v_threat_mean=20.0,
            v_threat_std=2.0,
            v_interceptor=22.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
        ),
    )

    # -------------------------------------------------
    # Scenario C: Clustered threats
    # -------------------------------------------------
    scenarios["C_clustered"] = CanonicalScenario(
        name="C_clustered",
        description="Threats appear in spatial clusters (low y variance).",
        params=ScenarioParams(
            seed=0,
            horizon_T=60.0,
            dt=0.5,
            lambda_arrival=0.20,
            x_spawn_mean=40.0,
            x_spawn_std=5.0,
            y_spawn_sigma=8.0,
            v_threat_mean=15.0,
            v_threat_std=2.0,
            v_interceptor=24.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
        ),
    )

    # -------------------------------------------------
    # Scenario D: Conflict (key scenario)
    # -------------------------------------------------
    scenarios["D_conflict"] = CanonicalScenario(
        name="D_conflict",
        description="Conflict between proximity (NI) and urgency (MPS/TTB).",
        params=ScenarioParams(
            seed=0,
            horizon_T=60.0,
            dt=0.5,
            lambda_arrival=0.12,
            x_spawn_mean=30.0,
            x_spawn_std=10.0,
            y_spawn_sigma=25.0,
            v_threat_mean=18.0,
            v_threat_std=3.0,
            v_interceptor=20.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
        ),
    )

    return scenarios


def get_stage1_heuristics():
    """
    Reduced set of heuristics for initial analysis.
    """
    return ["NI", "TTB", "MPS", "Weighted", "Cluster", "Random"]
