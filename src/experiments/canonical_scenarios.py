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

    # 1. NI vs MPS conflict
    scenarios["manual_conflict_ni_vs_mps"] = CanonicalScenario(
        name="manual_conflict_ni_vs_mps",
        description="One target is closer to the interceptor, another is more urgent near the boundary.",
        params=ScenarioParams(
            seed=0,
            horizon_T=20.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=20.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [25.0, 5.0], "vel": [-15.0, 0.0]},
                {"t": 0.0, "pos": [10.0, 30.0], "vel": [-20.0, 0.0]},
            ],
        ),
    )

    # 2. Cluster scenario
    scenarios["manual_cluster"] = CanonicalScenario(
        name="manual_cluster",
        description="Three threats are clustered, one threat is isolated.",
        params=ScenarioParams(
            seed=0,
            horizon_T=25.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=20.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [40.0, 5.0], "vel": [-15.0, 0.0]},
                {"t": 0.0, "pos": [42.0, 8.0], "vel": [-15.0, 0.0]},
                {"t": 0.0, "pos": [38.0, 3.0], "vel": [-15.0, 0.0]},
                {"t": 0.0, "pos": [25.0, -30.0], "vel": [-15.0, 0.0]},
            ],
        ),
    )

    # 3. Lost target + future value
    scenarios["manual_lost_but_repositioning_value"] = CanonicalScenario(
        name="manual_lost_but_repositioning_value",
        description="First target is likely lost, but moving toward it may help with future threats in the same area.",
        params=ScenarioParams(
            seed=0,
            horizon_T=25.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=20.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [5.0, 20.0], "vel": [-25.0, 0.0]},
                {"t": 2.0, "pos": [30.0, 20.0], "vel": [-15.0, 0.0]},
                {"t": 3.0, "pos": [32.0, 18.0], "vel": [-15.0, 0.0]},
            ],
        ),
    )

    # 4. Small-load conflict over time
    scenarios["manual_small_load_conflict"] = CanonicalScenario(
        name="manual_small_load_conflict",
        description="Three threats arrive over time and create repeated tradeoffs.",
        params=ScenarioParams(
            seed=0,
            horizon_T=25.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=20.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [35.0, 10.0], "vel": [-15.0, 0.0]},
                {"t": 1.0, "pos": [30.0, -20.0], "vel": [-15.0, 0.0]},
                {"t": 2.0, "pos": [25.0, 25.0], "vel": [-18.0, 0.0]},
            ],
        ),
    )

    return scenarios


def get_stage1_heuristics():
    return ["NI", "TTB", "MPS", "Weighted", "Cluster", "Random"]
