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
