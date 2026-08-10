# Decision-epoch relabeling and conservative closed-loop μFC

This revision fixes two issues exposed by the first closed-loop smoke test:

1. **Training-state sampling:** the historical labeling loop sampled the first active integration steps, not genuine target-selection epochs. The corrected sampler follows the full behavior trajectory, records states only when a new pursued target must be selected, and spreads the requested samples over the beginning, middle, and end of the trajectory.
2. **Small or invalid overrides:** the adaptive selector excludes heuristics that do not currently propose a target and can retain NT unless an alternative reduces predicted regret by a configurable threshold.

The labels remain **fixed-continuation labels**. This is not adaptive relabeling or policy iteration. The immediate experiment is: train μFC from corrected decision-epoch states, then invoke it again after each pursued target is intercepted or crosses the boundary.

## 1. Tests

```bash
python -m pytest -q tests/test_closed_loop_fc_selector.py
```

## 2. Generate a corrected pilot dataset

```bash
python -m src.experiments.run_large_scale_rollout \
  --n-scenarios 500 \
  --state-label-scenarios 500 \
  --scenario-mix decision_rich \
  --max-states-per-run 4 \
  --state-sampling-mode decision_epochs_uniform \
  --behavior-no-target-fallback NI \
  --skip-full-rollouts \
  --output-dir outputs/decision_epoch_labels_500
```

The important dataset is:

```text
outputs/decision_epoch_labels_500/large_scale_rollout_states_informative_no_ties.csv
```

Quick coverage check:

```python
import pandas as pd

path = "outputs/decision_epoch_labels_500/large_scale_rollout_states_informative_no_ties.csv"
df = pd.read_csv(path)
print(df[["t", "remaining_horizon", "behavior_decision_index", "behavior_decision_count"]].describe())
print(df["decision_epoch_reason"].value_counts())
print(df["behavior_fallback_used"].value_counts())
```

The time values should span the trajectory instead of being concentrated near `t=0`.

## 3. Evaluate with a manually selected regret threshold

A positive threshold keeps NT unless the best valid alternative is predicted to improve regret by at least that amount.

```bash
python -m src.experiments.run_closed_loop_fc_selector_experiment \
  --rollout-input-dir outputs/decision_epoch_labels_500 \
  --output-dir outputs/closed_loop_decision_epoch_pilot \
  --dataset-mode no_ties \
  --scenario-mix decision_rich \
  --n-estimators 200 \
  --min-samples-leaf 5 \
  --threshold-mode nt_override \
  --regret-threshold 1.0 \
  --n-test-scenarios 100 \
  --test-seed 20260810 \
  --bootstrap-samples 2000
```

## 4. Recommended: select the threshold on a separate validation set

```bash
python -m src.experiments.run_closed_loop_fc_selector_experiment \
  --rollout-input-dir outputs/decision_epoch_labels_500 \
  --output-dir outputs/closed_loop_decision_epoch_tuned \
  --dataset-mode no_ties \
  --scenario-mix decision_rich \
  --n-estimators 200 \
  --min-samples-leaf 5 \
  --threshold-mode nt_override \
  --threshold-grid 0,0.25,0.5,1,1.5,2,3 \
  --n-threshold-validation-scenarios 60 \
  --threshold-validation-seed 20260809 \
  --n-test-scenarios 100 \
  --test-seed 20260810 \
  --bootstrap-samples 2000
```

The threshold is selected only on the validation scenarios. The final test scenarios use a different seed and remain untouched during tuning.

## Threshold modes

- `nt_override` — recommended. NT is the baseline at every decision epoch; another valid heuristic is selected only if its predicted regret is lower by at least the threshold.
- `previous` — hysteresis. Retain the previous heuristic unless the current best is better by at least the threshold. If the previous heuristic is invalid, the baseline falls back to NT.
- `none` — no threshold. Choose the valid heuristic with minimum predicted regret.

## Main outputs

- `closed_loop_policy_summary.csv`
- `closed_loop_paired_comparison_vs_always_nt.csv`
- `adaptive_mu_fc_decision_log.csv`
- `adaptive_mu_fc_heuristic_usage.csv`
- `closed_loop_threshold_validation.csv` when a threshold grid is used
- `mu_fc_selector_configured.joblib`

The decision log records the unconstrained best heuristic, threshold baseline, predicted improvement, whether the threshold blocked an override, the valid-heuristic count, and the target proposed by every heuristic.
