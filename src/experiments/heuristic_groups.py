from __future__ import annotations

# Coarse grouping used by older exploratory group-classification experiments.
# The main proposal-level portfolio is:
# NI, FNI, FMTTB, MPS, FCluster.

HEURISTIC_TO_GROUP = {
    "NI": "Proximity",
    "FNI": "FeasibleProximity",
    "FMTTB": "FeasibleUrgency",
    "MPS": "FeasibleSlack",
    "FCluster": "SpatialPositioning",
}
