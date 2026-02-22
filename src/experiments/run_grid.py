import pandas as pd

from src.sim.env import ScenarioParams
from src.sim.runner import run_episode
from src.sim.heuristics import make_heuristics


def choose_winner(stats_df: pd.DataFrame) -> pd.Series:
    """
    Choose winner heuristic based on:
      1) max mean_intercepted
      2) min mean_escaped
      3) max mean_spawned
      4) min std_intercepted (prefer more stable)
    """
    df = stats_df.copy()

    df = df.sort_values(
        by=["mean_intercepted", "mean_escaped", "mean_spawned", "std_intercepted"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    return df.iloc[0]


def main():
    """
    Multi-seed grid runner:
      - For each scenario parameter set (excluding seed), run N seeds
      - For each heuristic, compute mean/std of intercepted/escaped/spawned
      - Pick winner by mean performance
      - Save:
          1) scenario_labels_multiseed.csv (one row per scenario with winner + top3)
          2) scenario_heuristic_stats.csv (one row per scenario x heuristic with stats)
    """

    # ===== SETTINGS =====
    N_SEEDS = 10
    base_seeds = list(range(N_SEEDS))

    # Expanded ranges (you can tune later)
    lambdas = [0.06, 0.12, 0.20, 0.30, 0.45]
    vI_list = [18.0, 22.0, 28.0]
    v_threat_means = [10.0, 14.0, 18.0, 22.0]
    v_threat_stds = [2.0, 6.0]
    x_spawn_means = [60.0, 40.0, 30.0, 20.0]
    x_spawn_stds = [8.0, 15.0]
    y_spawn_sigmas = [25.0, 60.0, 100.0]
    kill_radii = [2.0]
    horizon_T = 120.0

    # Heuristics available in the codebase (names must match make_heuristics keys)
    heuristic_names = list(make_heuristics(seed=0).keys())

    # Which heuristics also run with preemption variant
    preempt_variants = set(["NI", "MPS"])  # keep simple for now

    # ===== BUILD SCENARIO GRID (seed-free) =====
    scenario_specs = []
    sid = 0
    for lam in lambdas:
        for vI in vI_list:
            for vt_mean in v_threat_means:
                for vt_std in v_threat_stds:
                    for x_mean in x_spawn_means:
                        for x_std in x_spawn_stds:
                            for y_sig in y_spawn_sigmas:
                                for rk in kill_radii:
                                    scenario_specs.append({
                                        "scenario_id": sid,
                                        "lambda_arrival": lam,
                                        "v_interceptor": vI,
                                        "v_threat_mean": vt_mean,
                                        "v_threat_std": vt_std,
                                        "x_spawn_mean": x_mean,
                                        "x_spawn_std": x_std,
                                        "y_spawn_sigma": y_sig,
                                        "kill_radius": rk,
                                        "horizon_T": horizon_T,
                                    })
                                    sid += 1

    print(f"Total scenario parameter sets (seed-free): {len(scenario_specs)}")
    print(f"Seeds per scenario: {N_SEEDS}")

    # ===== RUN MULTI-SEED EVAL =====
    all_stats_rows = []    # scenario x heuristic rows (aggregated)
    all_detail_rows = []   # (optional) per-run rows, can be huge; we won't save by default

    label_rows = []

    total = len(scenario_specs)
    for idx, spec in enumerate(scenario_specs, start=1):
        scenario_id = spec["scenario_id"]

        # collect per-heuristic per-seed outcomes
        per_h_rows = []

        for hname in heuristic_names:
            for preempt in ([False, True] if hname in preempt_variants else [False]):

                intercepted_list = []
                escaped_list = []
                spawned_list = []

                for seed in base_seeds:
                    sp = ScenarioParams(
                        seed=seed,
                        horizon_T=spec["horizon_T"],

                        lambda_arrival=spec["lambda_arrival"],

                        x_spawn_mean=spec["x_spawn_mean"],
                        x_spawn_std=spec["x_spawn_std"],
                        y_spawn_sigma=spec["y_spawn_sigma"],

                        v_threat_mean=spec["v_threat_mean"],
                        v_threat_std=spec["v_threat_std"],

                        v_interceptor=spec["v_interceptor"],
                        kill_radius=spec["kill_radius"],
                    )

                    out = run_episode(sp, hname, preempt=preempt)
                    intercepted_list.append(out["intercepted"])
                    escaped_list.append(out["escaped"])
                    spawned_list.append(out["spawned"])

                # aggregate
                row = {
                    "scenario_id": scenario_id,
                    "heuristic": hname,
                    "preempt": preempt,

                    # scenario params
                    "lambda_arrival": spec["lambda_arrival"],
                    "x_spawn_mean": spec["x_spawn_mean"],
                    "x_spawn_std": spec["x_spawn_std"],
                    "y_spawn_sigma": spec["y_spawn_sigma"],
                    "v_threat_mean": spec["v_threat_mean"],
                    "v_threat_std": spec["v_threat_std"],
                    "v_interceptor": spec["v_interceptor"],
                    "kill_radius": spec["kill_radius"],
                    "horizon_T": spec["horizon_T"],

                    "n_seeds": N_SEEDS,

                    # aggregated outcomes
                    "mean_intercepted": float(pd.Series(intercepted_list).mean()),
                    "std_intercepted": float(pd.Series(intercepted_list).std(ddof=1)) if N_SEEDS > 1 else 0.0,
                    "mean_escaped": float(pd.Series(escaped_list).mean()),
                    "std_escaped": float(pd.Series(escaped_list).std(ddof=1)) if N_SEEDS > 1 else 0.0,
                    "mean_spawned": float(pd.Series(spawned_list).mean()),
                    "std_spawned": float(pd.Series(spawned_list).std(ddof=1)) if N_SEEDS > 1 else 0.0,
                }

                per_h_rows.append(row)

        stats_df = pd.DataFrame(per_h_rows)

        # winner selection
        winner_row = choose_winner(stats_df)

        # top-3 heuristics
        top3_df = stats_df.sort_values(
            by=["mean_intercepted", "mean_escaped", "mean_spawned", "std_intercepted"],
            ascending=[False, True, False, True],
        ).head(3)

        top3 = top3_df[[
            "heuristic", "preempt",
            "mean_intercepted", "mean_escaped", "mean_spawned",
            "std_intercepted"
        ]].to_dict("records")

        label_rows.append({
            "scenario_id": scenario_id,

            # scenario params
            "lambda_arrival": spec["lambda_arrival"],
            "x_spawn_mean": spec["x_spawn_mean"],
            "x_spawn_std": spec["x_spawn_std"],
            "y_spawn_sigma": spec["y_spawn_sigma"],
            "v_threat_mean": spec["v_threat_mean"],
            "v_threat_std": spec["v_threat_std"],
            "v_interceptor": spec["v_interceptor"],
            "kill_radius": spec["kill_radius"],
            "horizon_T": spec["horizon_T"],

            "n_seeds": N_SEEDS,

            # label
            "winner": f"{winner_row['heuristic']}{'+P' if bool(winner_row['preempt']) else ''}",
            "winner_mean_intercepted": float(winner_row["mean_intercepted"]),
            "winner_mean_escaped": float(winner_row["mean_escaped"]),
            "winner_std_intercepted": float(winner_row["std_intercepted"]),

            "top3": top3,
        })

        # accumulate all heuristic stats
        all_stats_rows.extend(per_h_rows)

        if idx % 20 == 0 or idx == total:
            print(f"Progress: {idx}/{total}")

    # ===== SAVE OUTPUTS =====
    labels_out = pd.DataFrame(label_rows)
    stats_out = pd.DataFrame(all_stats_rows)

    labels_out.to_csv("scenario_labels_multiseed.csv", index=False)
    stats_out.to_csv("scenario_heuristic_stats.csv", index=False)

    print("\nSaved: scenario_labels_multiseed.csv")
    print(labels_out.head())
    print("\nWinner counts:\n", labels_out["winner"].value_counts())

    print("\nSaved: scenario_heuristic_stats.csv")
    print(stats_out.head())


if __name__ == "__main__":
    main()
