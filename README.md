# dynamic-interception-sim
Dynamic interception sim of Ohad Nir
Dynamic interception simulator for evaluating fixed heuristics and learned heuristic selectors.

Current proposal-level heuristic portfolio:

- `NI` - nearest active target.
- `FNI` - nearest feasible target by time-to-intercept.
- `FMTTB` - feasible target with minimum time-to-boundary.
- `MPS` - feasible target with minimum positive slack.
- `FCluster` - leading-edge target of the densest local target neighborhood.

`Danger` and `Ratio` are not part of the current clean proposal portfolio.
