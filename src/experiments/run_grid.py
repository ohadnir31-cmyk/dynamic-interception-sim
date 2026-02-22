import pandas as pd

from src.sim.env import ScenarioParams
from src.sim.runner import compare_heuristics


def main():
    """
    Expanded scenario grid to create harder regimes where NI is not always best.
    Outputs:
      - scenario_labels.csv: one row per scenario with winner and top-3 heuristics
    """

    grid = []
    sid = 0

    # Expanded ranges (harder scenarios)
    seeds = [0, 1, 2]

    # Load (arrival intensity)
    lambdas = [0.06, 0.12, 0.20, 0.30, 0.45]

    # Interceptor capability
    vI_list = [18.0, 22.0, 28.0]

    # Threat speed regimes (mean + variability)
    v_threat_means = [10.0, 14.0, 18.0, 22.0]
    v_threat_stds = [2.0, 6.0]

    # Where threats spawn relative to the boundary (x=0)
    x_spawn_means = [60.0, 40.0, 30.0, 20.0]
    x_spawn_stds = [8.0, 15.0]

    # Lateral spread along the boundary
    y_spawn_sigmas = [25.0, 60.0, 100.0]

    # Kill radius & horizon (keep simple for now)
    kill_radii = [2.0]
    horizon_T = 120.0

    for seed in seeds:
        for lam in lambdas:
            for vI in vI_list:
                for vt_mean in v_threat_means:
                    for vt_std in v_threat_stds:
                        for x_mean in x_spawn_means:
                            for x_std in x_spawn_stds:
                                for y_sig in y_spawn_sigmas:
                                    for rk in kill_radii:
                                        sp = ScenarioParams(
                                            seed=seed,
                                            horizon_T=horizon_T,
                                            lambda_arrival=lam,
                                            v_interceptor=vI,
                                            v_threat_mean=vt_mean,
                                            v_threat_std=vt_std,
                                            x_spawn_mean=x_mean,
                                            x_spawn_std=x_std,
                                            y_spawn_sigma=y_sig,
                                            kill_radius=rk,
                                        )
                                        grid.append((sid, sp))
                                        sid += 1

    print(f"Total scenarios: {len(grid)}")

    label_rows = []
    for sid, sp in grid:
        df = compare_heuristics(sp)

        winner = df.iloc[0]["heuristic"]
        winner_intercepted = int(df.iloc[0]["intercepted"])
        winner_escaped = int(df.iloc[0]["escaped"])

        top3 = df.head(3)[["heuristic", "preempt", "intercepted", "escaped", "spawned"]].to_dict("records")

        label_rows.append({
            "scenario_id": sid,

            # scenario params (for now; later we'll also store decision-time features)
            "seed": sp.seed,
            "horizon_T": sp.horizon_T,
            "dt": sp.dt,
            "lambda_arrival": sp.lambda_arrival,

            "x_spawn_mean": sp.x_spawn_mean,
            "x_spawn_std": sp.x_spawn_std,
            "y_spawn_sigma": sp.y_spawn_sigma,

            "v_threat_mean": sp.v_threat_mean,
            "v_threat_std": sp.v_threat_std,

            "v_interceptor": sp.v_interceptor,
            "kill_radius": sp.kill_radius,

            # labels / outcomes
            "winner": winner,
            "winner_intercepted": winner_intercepted,
            "winner_escaped": winner_escaped,
            "top3": top3,
        })

        # lightweight progress
        if (sid + 1) % 50 == 0:
            print(f"Progress: {sid + 1}/{len(grid)}")

    out = pd.DataFrame(label_rows)
    out.to_csv("scenario_labels.csv", index=False)

    print("\nSaved: scenario_labels.csv")
    print(out.head())
    print("\nWinner counts:\n", out["winner"].value_counts())


if __name__ == "__main__":
    main()
