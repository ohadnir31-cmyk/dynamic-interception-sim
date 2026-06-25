# Chasing Repairman – Working Notes

## 1. Research Goal
The current goal is to build a trustworthy dynamic simulation environment, define canonical scenarios, and compare a small set of heuristics in regimes that can be visually understood and statistically analyzed.

## 2. World Model
- Boundary at x=0
- Threats spawn online or via manual scenarios
- Threats move toward the boundary
- Interceptor moves in 2D
- Escape occurs when threat crosses x<=0
- Intercept occurs when threat enters kill radius

## 3. Stage-1 Heuristics
- NI
- TTB
- MPS
- Weighted
- Cluster
- Random

## 4. Canonical Manual Scenarios
- manual_conflict_ni_vs_mps
- manual_cluster
- manual_lost_but_repositioning_value
- manual_small_load_conflict

## 5. First Questions
- When does NI fail?
- When does MPS outperform NI?
- Does Cluster help under true spatial concentration?
- How often are targets already lost at birth?

## 6. Next Extensions
- Decision-level logging for RL
- Weighted targets
- Splitting threats / submunitions
- Event-driven decision logic

#######################
# Working Notes - Dynamic Interception Sim
#######################
## Current proposal-level heuristic portfolio

The current clean portfolio is:

- NI - nearest active target by geometric distance.
- FNI - nearest feasible target by time-to-intercept.
- FMTTB - feasible target with minimum time-to-boundary.
- MPS - feasible target with minimum positive slack.
- FCluster - Frontier-Cluster: select the leading-edge target of the densest local target neighborhood.

Danger and Ratio were intentionally removed from the current proposal-level portfolio.
Composite priority / danger scores may be revisited in future work after heterogeneous target values are introduced.

## Frontier-Cluster definition

A local neighborhood radius is defined by interceptor mobility:

r = v_interceptor * cluster_time_window

The current default is cluster_time_window = 5.0.

For each target, define a local neighborhood as all active targets within radius r.
Choose the densest local neighborhood. If there is a tie, prefer the neighborhood whose leading target is closest to the protected boundary x = 0.
From the selected neighborhood, choose the target with the smallest x-coordinate.

## Main next analyses

- Run large-scale rollout labels using the clean portfolio.
- Evaluate observable-only winner classifier and observable-only regret selector.
- Compare learned selectors against fixed baselines.
- Produce representative diagnostic trace figures from real rollout states.
- Use future work to discuss regime-aware selectors, regret/ranking models, closed-loop adaptive deployment, and weighted targets.

######
# Working Notes - Dynamic Interception Sim
######
## Current proposal-level heuristic portfolio

The current clean portfolio is:

- NI - nearest active target by geometric distance.
- FNI - nearest feasible target by moving-target lead-intercept time.
- FMTTB - feasible target with minimum time-to-boundary.
- MPS - feasible target with minimum positive slack.
- FCluster - Frontier-Cluster: select the leading-edge target of the densest local target neighborhood.

Danger and Ratio were intentionally removed from the current proposal-level portfolio.
Composite priority / danger scores may be revisited in future work after heterogeneous target values are introduced.

## Moving-target interception model

The simulator and the heuristic feasibility calculations now use a moving-target lead-intercept time rather than the old static approximation of distance-to-current-position divided by interceptor speed.

For an interceptor at position `pI`, a target at position `pT`, target velocity `vT`, and interceptor speed `vI`, the time-to-intercept is the smallest non-negative solution of:

`||pT + vT * t - pI|| = vI * t`

Slack is then defined as:

`slack = TTB - TTI`

where TTB is the time-to-boundary and TTI is the moving-target lead-intercept time.

The simulator also uses lead-pursuit guidance: when a target is assigned, the interceptor steers toward the predicted future intercept point, recomputed at each time step for the currently assigned target.

## Frontier-Cluster definition

A local neighborhood radius is defined by interceptor mobility:

`r = v_interceptor * cluster_time_window`

The current default is `cluster_time_window = 5.0`.

For each target, define a local neighborhood as all active targets within radius `r`.
Choose the densest local neighborhood. If there is a tie, prefer the neighborhood whose leading target is closest to the protected boundary `x = 0`.
From the selected neighborhood, choose the target with the smallest x-coordinate.

## Main next analyses

- Re-run large-scale rollout labels using the clean portfolio and moving-target interception model.
- Evaluate observable-only winner classifier and observable-only regret selector.
- Compare learned selectors against fixed baselines.
- Produce representative diagnostic trace figures from real rollout states.
- Use future work to discuss regime-aware selectors, regret/ranking models, closed-loop adaptive deployment, and weighted targets.


# Working Notes - Dynamic Interception Sim

## Current proposal-level heuristic portfolio

The current clean portfolio is:

- NI - nearest active target by geometric distance.
- FNI - nearest feasible target by moving-target lead-intercept time.
- FMTTB - feasible target with minimum time-to-boundary.
- MPS - feasible target with minimum positive slack.
- FCluster - Frontier-Cluster: select the leading-edge target of the densest local target neighborhood.

Danger and Ratio were intentionally removed from the current proposal-level portfolio.
Composite priority / danger scores may be revisited in future work after heterogeneous target values are introduced.

## Moving-target interception model

The simulator and the heuristic feasibility calculations now use a moving-target lead-intercept time rather than the old static approximation of distance-to-current-position divided by interceptor speed.

For an interceptor at position `pI`, a target at position `pT`, target velocity `vT`, and interceptor speed `vI`, the time-to-intercept is the smallest non-negative solution of:

`||pT + vT * t - pI|| = vI * t`

Slack is then defined as:

`slack = TTB - TTI`

where TTB is the time-to-boundary and TTI is the moving-target lead-intercept time.

The simulator also uses lead-pursuit guidance: when a target is assigned, the interceptor steers toward the predicted future intercept point, recomputed at each time step for the currently assigned target.

## Frontier-Cluster definition

A local neighborhood radius is defined by interceptor mobility:

`r = v_interceptor * cluster_time_window`

The current default is `cluster_time_window = 5.0`.

For each target, define a local neighborhood as all active targets within radius `r`.
Choose the densest local neighborhood. If there is a tie, prefer the neighborhood whose leading target is closest to the protected boundary `x = 0`.
From the selected neighborhood, choose the target with the smallest x-coordinate.

## Main next analyses

- Re-run large-scale rollout labels using the clean portfolio and moving-target interception model.
- Evaluate observable-only winner classifier and observable-only regret selector.
- Compare learned selectors against fixed baselines.
- Produce representative diagnostic trace figures from real rollout states.
- Use future work to discuss regime-aware selectors, regret/ranking models, closed-loop adaptive deployment, and weighted targets.

## Scenario-mix update: decision-rich experiments

The scenario generator now supports `--scenario-mix` with three options:

- `baseline`: original broad distribution.
- `decision_rich`: emphasizes medium/high-load states without letting overloaded cases dominate completely.
- `heavy_load`: stronger stress-test distribution.

Recommended next run:

```bash
python -m src.experiments.run_large_scale_rollout \
  --n-scenarios 1000 \
  --state-label-scenarios 300 \
  --max-states-per-run 4 \
  --scenario-mix decision_rich \
  --output-dir /content/drive/MyDrive/dynamic_interception_outputs/large_scale_1000_lead_pursuit_decision_rich
```

The purpose is not to artificially disadvantage NI, but to create more decision-rich states where proximity, urgency, feasibility, and spatial density may disagree. This should make it easier to identify regimes in which adaptive selection has value.

## NI win/loss analysis plan

The anchored-NI selector is not the current focus. Instead, the current diagnostic goal is to understand when NI wins or loses.

Use:

```bash
python -m src.experiments.analyze_ni_win_loss \
  --input-dir /content/drive/MyDrive/dynamic_interception_outputs/large_scale_1000_lead_pursuit_decision_rich \
  --dataset-mode no_ties \
  --strong-loss-threshold 2
```

Key outputs:

- `ni_win_loss_overall.csv` - overall NI win/loss rates and mean regret.
- `ni_loss_by_N_active_bucket.csv` - active-target regimes where NI loses more often.
- `ni_loss_by_scenario_regime.csv` - load-regime differences.
- `winner_distribution_when_NI_loses.csv` - which heuristic usually beats NI.
- `ni_loss_feature_comparison.csv` - features that differ between NI-best and NI-loss states.

Interpretation focus:

- Does NI lose mostly in medium/high-load states rather than very low-load or overloaded states?
- When NI loses, is the winner usually FNI, FMTTB, MPS, or FCluster?
- Are NI-loss states characterized by higher spatial dispersion, lower feasible ratio, smaller slack, or stronger clustering?

# Working Notes - Dynamic Interception Sim

## Current proposal-level heuristic portfolio

The current clean portfolio is:

- NI - nearest active target by geometric distance.
- FNI - nearest feasible target by moving-target lead-intercept time.
- FMTTB - feasible target with minimum time-to-boundary.
- MPS - feasible target with minimum positive slack.
- FCluster - Frontier-Cluster: select the leading-edge target of the densest local target neighborhood.

Danger and Ratio were intentionally removed from the current proposal-level portfolio.
Composite priority / danger scores may be revisited in future work after heterogeneous target values are introduced.

## Moving-target interception model

The simulator and the heuristic feasibility calculations now use a moving-target lead-intercept time rather than the old static approximation of distance-to-current-position divided by interceptor speed.

For an interceptor at position `pI`, a target at position `pT`, target velocity `vT`, and interceptor speed `vI`, the time-to-intercept is the smallest non-negative solution of:

`||pT + vT * t - pI|| = vI * t`

Slack is then defined as:

`slack = TTB - TTI`

where TTB is the time-to-boundary and TTI is the moving-target lead-intercept time.

The simulator also uses lead-pursuit guidance: when a target is assigned, the interceptor steers toward the predicted future intercept point, recomputed at each time step for the currently assigned target.

## Frontier-Cluster definition

A local neighborhood radius is defined by interceptor mobility:

`r = v_interceptor * cluster_time_window`

The current default is `cluster_time_window = 5.0`.

For each target, define a local neighborhood as all active targets within radius `r`.
Choose the densest local neighborhood. If there is a tie, prefer the neighborhood whose leading target is closest to the protected boundary `x = 0`.
From the selected neighborhood, choose the target with the smallest x-coordinate.

## Main next analyses

- Re-run large-scale rollout labels using the clean portfolio and moving-target interception model.
- Evaluate observable-only winner classifier and observable-only regret selector.
- Compare learned selectors against fixed baselines.
- Produce representative diagnostic trace figures from real rollout states.
- Use future work to discuss regime-aware selectors, regret/ranking models, closed-loop adaptive deployment, and weighted targets.

## Scenario-mix update: decision-rich experiments

The scenario generator now supports `--scenario-mix` with three options:

- `baseline`: original broad distribution.
- `decision_rich`: emphasizes medium/high-load states without letting overloaded cases dominate completely.
- `heavy_load`: stronger stress-test distribution.

Recommended next run:

```bash
python -m src.experiments.run_large_scale_rollout \
  --n-scenarios 1000 \
  --state-label-scenarios 300 \
  --max-states-per-run 4 \
  --scenario-mix decision_rich \
  --output-dir /content/drive/MyDrive/dynamic_interception_outputs/large_scale_1000_lead_pursuit_decision_rich
```

The purpose is not to artificially disadvantage NI, but to create more decision-rich states where proximity, urgency, feasibility, and spatial density may disagree. This should make it easier to identify regimes in which adaptive selection has value.

## NI win/loss analysis plan

The anchored-NI selector is not the current focus. Instead, the current diagnostic goal is to understand when NI wins or loses.

Use:

```bash
python -m src.experiments.analyze_ni_win_loss \
  --input-dir /content/drive/MyDrive/dynamic_interception_outputs/large_scale_1000_lead_pursuit_decision_rich \
  --dataset-mode no_ties \
  --strong-loss-threshold 2
```

Key outputs:

- `ni_win_loss_overall.csv` - overall NI win/loss rates and mean regret.
- `ni_loss_by_N_active_bucket.csv` - active-target regimes where NI loses more often.
- `ni_loss_by_scenario_regime.csv` - load-regime differences.
- `winner_distribution_when_NI_loses.csv` - which heuristic usually beats NI.
- `ni_loss_feature_comparison.csv` - features that differ between NI-best and NI-loss states.

Interpretation focus:

- Does NI lose mostly in medium/high-load states rather than very low-load or overloaded states?
- When NI loses, is the winner usually FNI, FMTTB, MPS, or FCluster?
- Are NI-loss states characterized by higher spatial dispersion, lower feasible ratio, smaller slack, or stronger clustering?

## Finite-horizon TTB/slack feature correction

A diagnostic check on the overnight decision-rich run found a small number of scenarios in which stochastic target-speed sampling produced targets with a boundary-direction speed close to zero. These targets had mathematically valid but operationally uninformative time-to-boundary values in the hundreds of thousands or millions. Although this affected only a small number of state rows, it could distort mean and standard-deviation features such as `mean_ttb`, `std_ttb`, `mean_slack`, and `std_slack`.

The simulation code was updated in two complementary ways:

1. Stochastic target generation now enforces minimum target speed and minimum boundary-direction speed through `ScenarioParams.min_threat_speed` and `ScenarioParams.min_boundary_speed`.
2. State-feature extraction now uses finite-horizon bounded values for TTB and slack statistics: `effective_ttb = min(raw_ttb, remaining_horizon)` and `effective_slack = effective_ttb - TTI`.

This correction keeps the mathematical definition of raw time-to-boundary available in the simulator while ensuring that learning and diagnostic features reflect the finite-horizon operational decision problem.
