# dynamic-interception-sim

Dynamic interception simulator for evaluating fixed heuristics and learned heuristic selectors.

Current proposal-level heuristic portfolio:

- `NI` - nearest active target by geometric distance.
- `FNI` - nearest feasible target by moving-target lead-intercept time.
- `FMTTB` - feasible target with minimum time-to-boundary.
- `MPS` - feasible target with minimum positive slack.
- `FCluster` - leading-edge target of the densest local target neighborhood.

`Danger` and `Ratio` are not part of the current clean proposal portfolio.

## Interception model

The simulator now uses a lead-pursuit interception model. For a selected moving target, the interceptor aims toward the predicted future intercept point rather than toward the target's current position. The time-to-intercept (TTI) used by feasibility, slack, and heuristic logic is computed by solving the moving-target intercept equation:

`||target_pos + target_vel * t - interceptor_pos|| = v_interceptor * t`

This makes the feasibility definition more consistent with the simulator dynamics:

`slack = time_to_boundary - moving_target_time_to_intercept`

# dynamic-interception-sim

Dynamic interception simulator for evaluating fixed heuristics and learned heuristic selectors.

Current proposal-level heuristic portfolio:

- `NI` - nearest active target by geometric distance.
- `FNI` - nearest feasible target by moving-target lead-intercept time.
- `FMTTB` - feasible target with minimum time-to-boundary.
- `MPS` - feasible target with minimum positive slack.
- `FCluster` - leading-edge target of the densest local target neighborhood.

`Danger` and `Ratio` are not part of the current clean proposal portfolio.

## Interception model

The simulator now uses a lead-pursuit interception model. For a selected moving target, the interceptor aims toward the predicted future intercept point rather than toward the target's current position. The time-to-intercept (TTI) used by feasibility, slack, and heuristic logic is computed by solving the moving-target intercept equation:

`||target_pos + target_vel * t - interceptor_pos|| = v_interceptor * t`

This makes the feasibility definition more consistent with the simulator dynamics:

`slack = time_to_boundary - moving_target_time_to_intercept`

## Large-scale scenario mixes

`run_large_scale_rollout.py` supports multiple scenario mixtures through `--scenario-mix`:

- `baseline` - the broad original mixture used in the clean lead-pursuit experiments.
- `decision_rich` - increases medium/high-load regimes, where nearest-first decisions are less likely to be uniformly optimal and adaptive heuristic selection is expected to be more informative.
- `heavy_load` - a heavier stress-test mixture for sensitivity checks.

Example decision-rich run:

```bash
python -m src.experiments.run_large_scale_rollout \
  --n-scenarios 1000 \
  --state-label-scenarios 300 \
  --max-states-per-run 4 \
  --scenario-mix decision_rich \
  --output-dir /content/drive/MyDrive/dynamic_interception_outputs/large_scale_1000_lead_pursuit_decision_rich
```

The baseline and decision-rich outputs should be kept in separate directories so their behavior can be compared directly.

## NI win/loss analysis

To understand when the nearest-intercept heuristic (`NI`) is strong and when it fails, run:

```bash
python -m src.experiments.analyze_ni_win_loss \
  --input-dir /content/drive/MyDrive/dynamic_interception_outputs/large_scale_1000_lead_pursuit_decision_rich \
  --dataset-mode no_ties \
  --strong-loss-threshold 2
```

This produces summaries by active-target bucket, scenario regime, spatial structure, arrival process, and deadline pressure, as well as feature comparisons between NI-best and NI-loss states.

# dynamic-interception-sim

Dynamic interception simulator for evaluating fixed heuristics and learned heuristic selectors.

Current proposal-level heuristic portfolio:

- `NI` - nearest active target by geometric distance.
- `FNI` - nearest feasible target by moving-target lead-intercept time.
- `FMTTB` - feasible target with minimum time-to-boundary.
- `MPS` - feasible target with minimum positive slack.
- `FCluster` - leading-edge target of the densest local target neighborhood.

`Danger` and `Ratio` are not part of the current clean proposal portfolio.

## Interception model

The simulator now uses a lead-pursuit interception model. For a selected moving target, the interceptor aims toward the predicted future intercept point rather than toward the target's current position. The time-to-intercept (TTI) used by feasibility, slack, and heuristic logic is computed by solving the moving-target intercept equation:

`||target_pos + target_vel * t - interceptor_pos|| = v_interceptor * t`

This makes the feasibility definition more consistent with the simulator dynamics:

`slack = time_to_boundary - moving_target_time_to_intercept`

## Large-scale scenario mixes

`run_large_scale_rollout.py` supports multiple scenario mixtures through `--scenario-mix`:

- `baseline` - the broad original mixture used in the clean lead-pursuit experiments.
- `decision_rich` - increases medium/high-load regimes, where nearest-first decisions are less likely to be uniformly optimal and adaptive heuristic selection is expected to be more informative.
- `heavy_load` - a heavier stress-test mixture for sensitivity checks.

Example decision-rich run:

```bash
python -m src.experiments.run_large_scale_rollout \
  --n-scenarios 1000 \
  --state-label-scenarios 300 \
  --max-states-per-run 4 \
  --scenario-mix decision_rich \
  --output-dir /content/drive/MyDrive/dynamic_interception_outputs/large_scale_1000_lead_pursuit_decision_rich
```

The baseline and decision-rich outputs should be kept in separate directories so their behavior can be compared directly.

## NI win/loss analysis

To understand when the nearest-intercept heuristic (`NI`) is strong and when it fails, run:

```bash
python -m src.experiments.analyze_ni_win_loss \
  --input-dir /content/drive/MyDrive/dynamic_interception_outputs/large_scale_1000_lead_pursuit_decision_rich \
  --dataset-mode no_ties \
  --strong-loss-threshold 2
```

This produces summaries by active-target bucket, scenario regime, spatial structure, arrival process, and deadline pressure, as well as feature comparisons between NI-best and NI-loss states.

### Finite-horizon TTB/slack features

Recent versions include two safeguards against pathological time-to-boundary feature outliers:

- stochastic targets have minimum total speed and minimum boundary-direction speed;
- state features use finite-horizon bounded TTB/slack statistics for learning and diagnostics.

For large-scale runs, the defaults are:

```bash
--min-threat-speed 0.05
--min-boundary-speed 0.05
```

Example decision-rich run:

```bash
python -m src.experiments.run_large_scale_rollout \
  --n-scenarios 1000 \
  --state-label-scenarios 300 \
  --max-states-per-run 4 \
  --scenario-mix decision_rich \
  --min-threat-speed 0.05 \
  --min-boundary-speed 0.05 \
  --output-dir outputs/large_scale_1000_decision_rich_fixed_ttb
```
