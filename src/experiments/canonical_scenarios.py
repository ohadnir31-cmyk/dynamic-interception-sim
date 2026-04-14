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

    scenarios["manual_conflict_ni_vs_mps"] = CanonicalScenario(
        name="manual_conflict_ni_vs_mps",
        description="Stronger conflict between proximity and urgency with three overlapping targets.",
        params=ScenarioParams(
            seed=0,
            horizon_T=25.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=14.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                # closer to interceptor, less urgent
                {"t": 0.0, "pos": [26.0, 5.0], "vel": [-8.0, 0.0]},
                # farther in Euclidean distance but more urgent
                {"t": 0.0, "pos": [14.0, 20.0], "vel": [-12.0, 0.0]},
                # another target to keep overlap and force tradeoff
                {"t": 1.0, "pos": [24.0, -10.0], "vel": [-10.0, 0.0]},
            ],
        ),
    )
    scenarios["manual_cluster"] = CanonicalScenario(
        name="manual_cluster",
        description=(
            "Three threats are spatially clustered, while one threat is isolated. "
            "Designed to test whether Cluster / Weighted gain from moving toward dense areas."
        ),
        params=ScenarioParams(
            seed=0,
            horizon_T=25.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=18.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [38.0, 8.0], "vel": [-10.0, 0.0]},
                {"t": 0.5, "pos": [40.0, 10.0], "vel": [-10.0, 0.0]},
                {"t": 1.0, "pos": [36.0, 6.0], "vel": [-10.0, 0.0]},
                {"t": 0.0, "pos": [24.0, -28.0], "vel": [-10.0, 0.0]},
            ],
        ),
    )

    scenarios["manual_lost_but_repositioning_value"] = CanonicalScenario(
        name="manual_lost_but_repositioning_value",
        description=(
            "The first threat is likely lost, but moving toward its area may help "
            "intercept later threats arriving nearby."
        ),
        params=ScenarioParams(
            seed=0,
            horizon_T=25.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=18.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [6.0, 18.0], "vel": [-22.0, 0.0]},
                {"t": 2.0, "pos": [28.0, 18.0], "vel": [-11.0, 0.0]},
                {"t": 3.0, "pos": [30.0, 20.0], "vel": [-11.0, 0.0]},
                {"t": 4.0, "pos": [26.0, 16.0], "vel": [-11.0, 0.0]},
            ],
        ),
    )

    scenarios["manual_small_load_conflict"] = CanonicalScenario(
        name="manual_small_load_conflict",
        description=(
            "A small number of threats arrive over time, but enough overlap is created "
            "to generate repeated target-selection dilemmas."
        ),
        params=ScenarioParams(
            seed=0,
            horizon_T=25.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=18.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [34.0, 10.0], "vel": [-11.0, 0.0]},
                {"t": 1.0, "pos": [26.0, -18.0], "vel": [-13.0, 0.0]},
                {"t": 2.0, "pos": [30.0, 22.0], "vel": [-12.0, 0.0]},
                {"t": 3.0, "pos": [22.0, 4.0], "vel": [-14.0, 0.0]},
            ],
        ),
    )

    return scenarios


def get_stage1_heuristics():
    return ["NI", "TTB", "MPS", "Weighted", "Cluster", "Random"]
