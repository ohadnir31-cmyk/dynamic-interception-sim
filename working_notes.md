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
