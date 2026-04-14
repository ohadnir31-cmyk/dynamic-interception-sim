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
    # 1. NI should do well
    # -------------------------------------------------
    scenarios["manual_ni_favorable"] = CanonicalScenario(
        name="manual_ni_favorable",
        description=(
            "All threats are feasible and no target is dramatically more urgent than the others. "
            "Designed so that nearest-first should perform well."
        ),
        params=ScenarioParams(
            seed=0,
            horizon_T=25.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=20.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [18.0, 6.0], "vel": [-8.0, 0.0]},
                {"t": 0.5, "pos": [26.0, -8.0], "vel": [-8.0, 0.0]},
                {"t": 1.0, "pos": [30.0, 10.0], "vel": [-8.0, 0.0]},
            ],
        ),
    )

    # -------------------------------------------------
    # 2. True conflict: MPS should help
    # -------------------------------------------------
    scenarios["manual_conflict_ni_vs_mps"] = CanonicalScenario(
        name="manual_conflict_ni_vs_mps",
        description=(
            "A true conflict between proximity and urgency. "
            "One threat is closer to the interceptor, while another is more urgent."
        ),
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
                # farther from interceptor, but more urgent
                {"t": 0.0, "pos": [14.0, 20.0], "vel": [-12.0, 0.0]},
                # another target to maintain overlap
                {"t": 1.0, "pos": [24.0, -10.0], "vel": [-10.0, 0.0]},
            ],
        ),
    )

    # -------------------------------------------------
    # 3. TTB-favorable: urgency matters, but feasibility is still present
    # -------------------------------------------------
    scenarios["manual_ttb_favorable"] = CanonicalScenario(
        name="manual_ttb_favorable",
        description=(
            "One target has a clearly shorter time-to-boundary while still being feasible. "
            "Designed to favor urgency-driven policies."
        ),
        params=ScenarioParams(
            seed=0,
            horizon_T=22.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=16.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                # relatively close but not urgent
                {"t": 0.0, "pos": [24.0, 4.0], "vel": [-8.0, 0.0]},
                # more urgent, should push TTB-like behavior
                {"t": 0.0, "pos": [10.0, 18.0], "vel": [-10.0, 0.0]},
                # mild background pressure
                {"t": 1.0, "pos": [28.0, -14.0], "vel": [-8.0, 0.0]},
            ],
        ),
    )

    # -------------------------------------------------
    # 4. Cluster should help
    # -------------------------------------------------
    scenarios["manual_cluster_favorable"] = CanonicalScenario(
        name="manual_cluster_favorable",
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
                # clustered group
                {"t": 0.0, "pos": [38.0, 8.0], "vel": [-10.0, 0.0]},
                {"t": 0.5, "pos": [40.0, 10.0], "vel": [-10.0, 0.0]},
                {"t": 1.0, "pos": [36.0, 6.0], "vel": [-10.0, 0.0]},
                # isolated target
                {"t": 0.0, "pos": [24.0, -28.0], "vel": [-10.0, 0.0]},
            ],
        ),
    )

    # -------------------------------------------------
    # 5. Weighted / compromise regime
    # -------------------------------------------------
    scenarios["manual_weighted_tradeoff"] = CanonicalScenario(
        name="manual_weighted_tradeoff",
        description=(
            "A compromise regime where neither pure proximity nor pure urgency is obviously best. "
            "Designed to test weighted tradeoff behavior."
        ),
        params=ScenarioParams(
            seed=0,
            horizon_T=24.0,
            dt=0.5,
            lambda_arrival=0.0,
            v_interceptor=16.0,
            kill_radius=2.0,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [24.0, 8.0], "vel": [-9.0, 0.0]},
                {"t": 0.0, "pos": [16.0, 16.0], "vel": [-10.0, 0.0]},
                {"t": 1.0, "pos": [26.0, -6.0], "vel": [-9.5, 0.0]},
                {"t": 2.0, "pos": [20.0, -18.0], "vel": [-10.5, 0.0]},
            ],
        ),
    )

    # -------------------------------------------------
    # 6. Lost target but possible future spatial value
    # -------------------------------------------------
    scenarios["manual_lost_but_repositioning_value"] = CanonicalScenario(
        name="manual_lost_but_repositioning_value",
        description=(
            "The first threat is likely lost, but moving toward its area may help "
            "intercept later threats arriving nearby. Useful for motivating future repositioning logic."
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

    return scenarios


def get_stage1_heuristics():
    """
    Small heuristic set for the initial qualitative and quantitative analysis.
    """
    return ["NI", "TTB", "MPS", "Weighted", "Cluster", "Random"]
