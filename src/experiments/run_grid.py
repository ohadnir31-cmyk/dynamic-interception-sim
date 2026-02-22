import pandas as pd
from src.sim.env import ScenarioParams
from src.sim.runner import compare_heuristics

def main():
    grid = []
    sid = 0
    for seed in [0, 1, 2]:
        for lam in [0.06, 0.12, 0.20]:
            for vI in [20.0, 28.0]:
                for vt in [10.0, 14.0]:
                    for ys in [25.0, 60.0]:
                        grid.append((sid, ScenarioParams(
                            seed=seed,
                            lambda_arrival=lam,
                            v_interceptor=vI,
                            v_threat_mean=vt,
                            y_spawn_sigma=ys,
                        )))
                        sid += 1

    label_rows = []
    for sid, sp in grid:
        df = compare_heuristics(sp)
        winner = df.iloc[0]["heuristic"]
        label_rows.append({
            "scenario_id": sid,
            "seed": sp.seed,
            "lambda_arrival": sp.lambda_arrival,
            "v_interceptor": sp.v_interceptor,
            "v_threat_mean": sp.v_threat_mean,
            "y_spawn_sigma": sp.y_spawn_sigma,
            "winner": winner,
            "winner_intercepted": int(df.iloc[0]["intercepted"]),
            "top3": df.head(3)[["heuristic", "intercepted", "escaped", "preempt"]].to_dict("records"),
        })

    out = pd.DataFrame(label_rows)
    out.to_csv("scenario_labels.csv", index=False)
    print(out.head())
    print("\nWinner counts:\n", out["winner"].value_counts())

if __name__ == "__main__":
    main()
