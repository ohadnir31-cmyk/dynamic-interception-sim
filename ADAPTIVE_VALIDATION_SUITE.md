# Focused adaptive validation suite

This experiment is **additive** to the original 5,000-scenario proposal study.
It does not replace the historical fixed-continuation tables. It adds a focused
closed-loop test in which the learned selector is invoked again whenever the
currently pursued target is intercepted or crosses the boundary.

The workflow has three parts:

1. generate a new fixed-continuation training dataset at genuine decision epochs;
2. run a 100/250/500-scenario learning curve, validation-based baseline and
   threshold selection, and a held-out 300-scenario test;
3. compare the learned policy with an exact adaptive heuristic-portfolio oracle
   on a small nontrivial scenario suite.

## 1. Generate the 500-scenario decision-epoch dataset

The initial state is labeled only once per scenario. Later decision states from
all five behavior heuristics are retained. Tied-best heuristic outcomes are
kept in the generated `with_ties` file.

```bash
python -m src.experiments.run_large_scale_rollout \
  --n-scenarios 500 \
  --state-label-scenarios 500 \
  --scenario-mix decision_rich \
  --max-states-per-run 6 \
  --state-sampling-mode decision_epochs_uniform \
  --behavior-no-target-fallback NI \
  --skip-full-rollouts \
  --output-dir /content/drive/MyDrive/dynamic_interception_outputs/decision_epoch_labels_500
```

Expected upper bound before action deduplication:

```text
500 scenarios × 5 behavior heuristics × 6 sampled states × 5 candidates
= 75,000 counterfactual rollouts
```

## 2. Run the learning curve, validation, and held-out test

```bash
python -m src.experiments.run_adaptive_validation_suite \
  --rollout-input-dir /content/drive/MyDrive/dynamic_interception_outputs/decision_epoch_labels_500 \
  --output-dir /content/drive/MyDrive/dynamic_interception_outputs/adaptive_validation_suite_500 \
  --dataset-mode with_ties \
  --training-sizes 100,250,500 \
  --training-size-selection largest \
  --n-estimators 400 \
  --min-samples-leaf 5 \
  --n-validation-scenarios 150 \
  --validation-seed 20260809 \
  --threshold-grid 0,0.25,0.5,1,1.5,2,3 \
  --threshold-selection-rule lower_ci \
  --n-test-scenarios 300 \
  --test-seed 20260810 \
  --scenario-mix decision_rich \
  --bootstrap-samples 5000
```

The script:

- removes duplicate initial-state rows as a second safety check;
- trains nested models on 100, 250, and 500 independent scenarios;
- evaluates all model sizes on the same fresh validation scenarios;
- selects the strongest fixed validation baseline, rather than assuming NT;
- tunes a conservative regret threshold relative to that baseline;
- freezes the model, baseline, and threshold before the test;
- evaluates fixed heuristics, one-shot `mu_FC`, ungated closed-loop `mu_FC`,
  and conservative closed-loop `mu_FC` on 300 new scenarios.

Important outputs:

```text
learning_curve_summary.csv
validation_fixed_policy_summary.csv
threshold_validation_summary.csv
selected_mu_fc_ungated.joblib
selected_mu_fc_configured.joblib
test_policy_summary.csv
test_paired_comparison_vs_validation_baseline.csv
test_paired_comparison_vs_nt.csv
test_adaptive_vs_best_fixed_hindsight.csv
test_adaptive_heuristic_usage.csv
adaptive_validation_suite_manifest.json
```

## 3. Run the exact small-instance adaptive portfolio oracle

The exact oracle enumerates all distinct target actions proposed by the five
heuristics at every decision epoch. Heuristics proposing the same target share
one branch. The default generator rejects trivial cases and retains scenarios
in which the adaptive oracle exceeds the best fixed heuristic.

```bash
python -m src.experiments.run_small_exact_oracle_experiment \
  --model-in /content/drive/MyDrive/dynamic_interception_outputs/adaptive_validation_suite_500/selected_mu_fc_configured.joblib \
  --output-dir /content/drive/MyDrive/dynamic_interception_outputs/small_exact_oracle \
  --n-scenarios 12 \
  --seed 12345 \
  --min-targets 4 \
  --max-targets 6 \
  --horizon 8 \
  --max-decisions 7 \
  --max-search-nodes 250000
```

Important outputs:

```text
small_exact_oracle_results.csv
small_exact_oracle_summary.csv
small_exact_oracle_paths.json
small_exact_oracle_manifest.json
```

The oracle is exact **within the five-heuristic portfolio**. It is not an
unrestricted oracle over every active target.

## Automated checks

```bash
python -m pytest -q tests/test_closed_loop_fc_selector.py
```

Expected result for this update:

```text
10 passed
```
