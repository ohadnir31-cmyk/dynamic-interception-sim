# from __future__ import annotations

# HEURISTIC_TO_GROUP = {
#     # Feasibility-oriented policies
#     "MPS": "Feasible",
#     "FNI": "Feasible",
#     "FMTTB": "Feasible",

#     # Ratio / urgency / composite urgency policies
#     "Ratio": "RatioUrgency",
#     "Danger": "RatioUrgency",
#     "Lookahead": "RatioUrgency",

#     # Spatial policy
#     "Cluster": "Spatial",

#     # Proximity / weighted policies
#     "NI": "Proximity",
#     "Weighted": "Proximity",
# }


from __future__ import annotations

HEURISTIC_TO_GROUP = {
    "MPS": "Feasible",
    "FNI": "Feasible",

    "Ratio": "RatioUrgency",
    "Danger": "RatioUrgency",

    "Cluster": "Spatial",

    "NI": "Proximity",
}
