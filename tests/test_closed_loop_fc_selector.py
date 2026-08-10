from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.closed_loop_fc_selector import (
    BASE_OBSERVABLE_FEATURES,
    OPTIONAL_OBSERVABLE_FEATURES,
    FixedContinuationRegretSelector,
    FixedHeuristicSelector,
    advance_one_pursuit,
    run_closed_loop_selector,
    train_mu_fc_from_existing_dataset,
)
from src.sim.env import ScenarioParams, SimEnv


HEURISTICS = ["NI", "FNI", "FMTTB", "MPS", "FCluster"]


def make_tiny_rollout_dataset(path: Path) -> None:
    rows = []
    rng = np.random.default_rng(7)
    features = list(BASE_OBSERVABLE_FEATURES) + list(OPTIONAL_OBSERVABLE_FEATURES)

    for scenario_index in range(12):
        for state_index in range(3):
            row = {
                "scenario": f"s{scenario_index:02d}",
                "winner": HEURISTICS[(scenario_index + state_index) % len(HEURISTICS)],
                "N_active": 2 + ((scenario_index + state_index) % 8),
            }
            for feature in features:
                if feature == "N_active":
                    continue
                row[feature] = float(rng.normal())

            scores = {
                heuristic: float(10 - abs(i - ((scenario_index + state_index) % 5)))
                for i, heuristic in enumerate(HEURISTICS)
            }
            best = max(scores.values())
            ranked = sorted(HEURISTICS, key=lambda h: (-scores[h], h))
            for heuristic in HEURISTICS:
                row[f"{heuristic}_future_intercepted"] = scores[heuristic]
                row[f"{heuristic}_future_escaped"] = 20 - scores[heuristic]
                row[f"{heuristic}_regret"] = best - scores[heuristic]
                row[f"{heuristic}_rank"] = ranked.index(heuristic) + 1
            row["best_future_intercepted"] = best
            row["best_future_escaped"] = 20 - best
            rows.append(row)

    pd.DataFrame(rows).to_csv(
        path / "large_scale_rollout_states_informative_no_ties.csv",
        index=False,
    )


class ClosedLoopFCSelectorTests(unittest.TestCase):
    def test_train_save_and_load_mu_fc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_tiny_rollout_dataset(directory)
            artifacts = train_mu_fc_from_existing_dataset(
                directory,
                validation_size=0.25,
                random_state=3,
                n_estimators=8,
                min_samples_leaf=1,
            )
            self.assertEqual(set(artifacts.selector.candidate_heuristics), set(HEURISTICS))
            self.assertFalse(artifacts.validation_summary.empty)

            model_path = directory / "mu_fc.joblib"
            artifacts.selector.save(model_path, artifacts.metadata)
            loaded, metadata = FixedContinuationRegretSelector.load(model_path)
            self.assertEqual(list(loaded.feature_columns), list(artifacts.selector.feature_columns))
            self.assertEqual(metadata["label_semantics"], "fixed-continuation regret")

    def test_selected_target_crossing_ends_one_pursuit(self) -> None:
        params = ScenarioParams(
            horizon_T=3.0,
            dt=0.25,
            v_interceptor=0.05,
            kill_radius=0.01,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [0.2, 0.0], "vel": [-1.0, 0.0]},
                {"t": 0.0, "pos": [2.0, 1.0], "vel": [-0.05, 0.0]},
            ],
        )
        env = SimEnv(params)
        env._spawn_manual_threats_due()  # test setup: expose t=0 targets
        transition = advance_one_pursuit(env, "NI")
        self.assertEqual(transition.termination_reason, "selected_target_crossed")
        self.assertLess(transition.end_time, params.horizon_T)

    def test_closed_loop_reselects_after_target_resolution(self) -> None:
        params = ScenarioParams(
            horizon_T=5.0,
            dt=0.25,
            v_interceptor=1.0,
            kill_radius=0.15,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [0.7, 0.0], "vel": [-0.05, 0.0]},
                {"t": 0.0, "pos": [1.8, 0.5], "vel": [-0.05, 0.0]},
            ],
        )
        result = run_closed_loop_selector(
            params,
            FixedHeuristicSelector("NI"),
            collect_decisions=True,
        )
        self.assertGreaterEqual(result["num_decisions"], 2)
        reasons = {row["termination_reason"] for row in result["decision_log"]}
        self.assertTrue(
            reasons.intersection(
                {"selected_target_intercepted", "selected_target_crossed"}
            )
        )


if __name__ == "__main__":
    unittest.main()
