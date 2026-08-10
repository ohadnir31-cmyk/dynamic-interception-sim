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
    preprocess_rollout_dataset,
    run_closed_loop_selector,
    run_one_shot_selector,
    train_mu_fc_from_existing_dataset,
)
from src.experiments.adaptive_portfolio_oracle import (
    exact_adaptive_portfolio_oracle,
)
from src.experiments.rollout_labeling import collect_behavior_decision_snapshots
from src.sim.env import ScenarioParams, SimEnv


HEURISTICS = ["NI", "FNI", "FMTTB", "MPS", "FCluster"]


class ConstantPredictionModel:
    def __init__(self, value: float):
        self.value = float(value)
        self.n_jobs = 1

    def predict(self, x):
        return np.full(len(x), self.value, dtype=float)


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


    def test_regret_threshold_blocks_small_override_of_nt(self) -> None:
        params = ScenarioParams(
            horizon_T=3.0,
            dt=0.25,
            v_interceptor=1.0,
            kill_radius=0.15,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [1.0, 0.0], "vel": [-0.05, 0.0]},
            ],
        )
        env = SimEnv(params)
        env._spawn_manual_threats_due()
        models = {
            "NI": ConstantPredictionModel(1.0),
            "FNI": ConstantPredictionModel(0.8),
            "FMTTB": ConstantPredictionModel(3.0),
            "MPS": ConstantPredictionModel(4.0),
            "FCluster": ConstantPredictionModel(5.0),
        }
        selector = FixedContinuationRegretSelector(
            models=models,
            feature_columns=[],
            medians={},
            candidate_heuristics=HEURISTICS,
            regret_threshold=0.5,
            threshold_mode="nt_override",
        )
        details = selector.decision_details(env)
        self.assertEqual(details["best_unconstrained_heuristic"], "FNI")
        self.assertEqual(details["selected_heuristic"], "NI")
        self.assertTrue(details["threshold_blocked"])

        selector.regret_threshold = 0.1
        details = selector.decision_details(env)
        self.assertEqual(details["selected_heuristic"], "FNI")
        self.assertFalse(details["threshold_blocked"])

    def test_invalid_feasibility_heuristics_are_masked(self) -> None:
        params = ScenarioParams(
            horizon_T=2.0,
            dt=0.25,
            v_interceptor=0.1,
            kill_radius=0.01,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [0.2, 10.0], "vel": [-1.0, 0.0]},
            ],
        )
        env = SimEnv(params)
        env._spawn_manual_threats_due()
        models = {
            "NI": ConstantPredictionModel(5.0),
            "FNI": ConstantPredictionModel(0.0),
            "FMTTB": ConstantPredictionModel(0.1),
            "MPS": ConstantPredictionModel(0.2),
            "FCluster": ConstantPredictionModel(10.0),
        }
        selector = FixedContinuationRegretSelector(
            models=models,
            feature_columns=[],
            medians={},
            candidate_heuristics=HEURISTICS,
            regret_threshold=0.0,
            threshold_mode="none",
        )
        details = selector.decision_details(env)
        self.assertEqual(details["selected_heuristic"], "NI")
        self.assertEqual(details["valid_heuristics"], ["NI", "FCluster"])
        self.assertIsNotNone(details["selected_target_id"])

    def test_decision_snapshots_cover_late_trajectory(self) -> None:
        params = ScenarioParams(
            horizon_T=8.0,
            dt=0.25,
            v_interceptor=1.5,
            kill_radius=0.15,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [0.8, 0.0], "vel": [-0.02, 0.0]},
                {"t": 1.5, "pos": [1.0, 0.5], "vel": [-0.02, 0.0]},
                {"t": 3.0, "pos": [1.2, -0.5], "vel": [-0.02, 0.0]},
                {"t": 4.5, "pos": [1.0, 0.2], "vel": [-0.02, 0.0]},
            ],
        )
        snapshots = collect_behavior_decision_snapshots(
            params,
            "NI",
            no_target_fallback="NI",
        )
        times = [float(item["env_snapshot"].t) for item in snapshots]
        self.assertGreaterEqual(len(times), 3)
        self.assertGreater(max(times), 3.0)
        self.assertTrue(all(b >= a for a, b in zip(times, times[1:])))

    def test_initial_state_deduplication_keeps_one_row_per_scenario(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "scenario": "s1",
                    "decision_epoch_reason": "initial",
                    "behavior_heuristic": heuristic,
                    "t": 0.0,
                }
                for heuristic in HEURISTICS
            ]
            + [
                {
                    "scenario": "s1",
                    "decision_epoch_reason": "selected_target_intercepted",
                    "behavior_heuristic": "FNI",
                    "t": 2.0,
                }
            ]
        )
        prepared, report = preprocess_rollout_dataset(frame)
        self.assertEqual(len(prepared), 2)
        self.assertEqual(
            int((prepared["decision_epoch_reason"] == "initial").sum()),
            1,
        )
        self.assertEqual(int(report.iloc[0]["rows_removed"]), 4)

    def test_configurable_baseline_threshold_uses_fni(self) -> None:
        params = ScenarioParams(
            horizon_T=3.0,
            dt=0.25,
            v_interceptor=1.0,
            kill_radius=0.15,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [1.0, 0.0], "vel": [-0.05, 0.0]},
            ],
        )
        env = SimEnv(params)
        env._spawn_manual_threats_due()
        models = {
            "NI": ConstantPredictionModel(0.6),
            "FNI": ConstantPredictionModel(1.0),
            "FMTTB": ConstantPredictionModel(0.8),
            "MPS": ConstantPredictionModel(4.0),
            "FCluster": ConstantPredictionModel(5.0),
        }
        selector = FixedContinuationRegretSelector(
            models=models,
            feature_columns=[],
            medians={},
            candidate_heuristics=HEURISTICS,
            regret_threshold=0.5,
            threshold_mode="baseline_override",
            baseline_heuristic="FNI",
        )
        details = selector.decision_details(env)
        self.assertEqual(details["best_unconstrained_heuristic"], "NI")
        self.assertEqual(details["baseline_heuristic"], "FNI")
        self.assertEqual(details["selected_heuristic"], "FNI")
        self.assertTrue(details["threshold_blocked"])

    def test_one_shot_selector_keeps_one_heuristic(self) -> None:
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
        result = run_one_shot_selector(
            params,
            FixedHeuristicSelector("NI"),
            collect_decisions=True,
        )
        self.assertEqual(result["num_heuristic_switches"], 0)
        self.assertEqual(set(result["heuristic_counts"]), {"NT"})
        self.assertGreaterEqual(result["num_decisions"], 1)

    def test_exact_portfolio_oracle_is_not_worse_than_fixed_nt(self) -> None:
        params = ScenarioParams(
            horizon_T=6.0,
            dt=0.25,
            v_interceptor=1.0,
            kill_radius=0.15,
            home=(0.0, 0.0),
            manual_threats=[
                {"t": 0.0, "pos": [0.9, 0.0], "vel": [-0.03, 0.0]},
                {"t": 0.0, "pos": [1.4, 0.7], "vel": [-0.03, 0.0]},
                {"t": 0.0, "pos": [1.8, -0.5], "vel": [-0.03, 0.0]},
            ],
        )
        fixed = run_closed_loop_selector(
            params,
            FixedHeuristicSelector("NI"),
            collect_decisions=False,
        )
        oracle = exact_adaptive_portfolio_oracle(
            params,
            heuristic_names=HEURISTICS,
            max_decisions=4,
            max_nodes=10_000,
        )
        self.assertGreaterEqual(oracle.intercepted, fixed["intercepted"])
        self.assertGreaterEqual(oracle.stats.nodes, 1)



if __name__ == "__main__":
    unittest.main()
