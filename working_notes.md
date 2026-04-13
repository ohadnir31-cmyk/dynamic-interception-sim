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
