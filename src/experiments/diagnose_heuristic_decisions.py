from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.experiments.plot_actual_representative_trace_figures import replay_behavior_to_selected_state
from src.sim.env import predicted_intercept_point, slack, time_to_boundary_x0, time_to_intercept
from src.sim.heuristics import make_heuristics


HEURISTICS = ["NI", "FNI", "FMTTB", "MPS", "FCluster"]
EPS = 1e-9


def dataset_path(input_dir: Path, mode: str) -> Path:
    if mode == "no_ties":
        return input_dir / "large_scale_rollout_states_informative_no_ties.csv"
    if mode == "with_ties":
        return input_dir / "large_scale_rollout_states_informative_with_ties.csv"
    if mode == "all":
        return input_dir / "large_scale_rollout_states.csv"
    raise ValueError(f"Unknown dataset mode: {mode}")


def normalize_target_id(value: Optional[int]) -> int:
    return -1 if value is None else int(value)


def expected_decisions(active, pI: np.ndarray, vI: float) -> dict[str, Optional[int]]:
    if not active:
        return {h: None for h in HEURISTICS}

    rows = []
    for th in active:
        dist = float(np.linalg.norm(th.pos - pI))
        tti = float(time_to_intercept(pI, th.pos, vI, th.vel))
        ttb = float(time_to_boundary_x0(th.pos, th.vel))
        s = float(slack(pI, th, vI))
        rows.append((th, dist, tti, ttb, s))

    feasible = [item for item in rows if item[4] >= -EPS]

    out: dict[str, Optional[int]] = {}
    out["NI"] = min(rows, key=lambda item: item[1])[0].id

    if feasible:
        out["FNI"] = min(feasible, key=lambda item: item[2])[0].id
        out["FMTTB"] = min(feasible, key=lambda item: item[3])[0].id
        out["MPS"] = min(feasible, key=lambda item: item[4])[0].id
    else:
        out["FNI"] = None
        out["FMTTB"] = None
        out["MPS"] = None

    # FCluster is intentionally not reimplemented here as a correctness oracle.
    # We still record its selected target metrics below.
    out["FCluster"] = None
    return out


def target_metrics(active, pI: np.ndarray, vI: float, target_id: Optional[int]) -> dict[str, Any]:
    if target_id is None:
        return {
            "selected_exists": False,
            "selected_dist": np.nan,
            "selected_tti": np.nan,
            "selected_ttb": np.nan,
            "selected_slack": np.nan,
            "selected_feasible": False,
            "selected_lead_x": np.nan,
            "selected_lead_x_lt_0": False,
        }

    for th in active:
        if int(th.id) == int(target_id):
            lead = predicted_intercept_point(pI, th.pos, th.vel, vI)
            s = float(slack(pI, th, vI))
            return {
                "selected_exists": True,
                "selected_dist": float(np.linalg.norm(th.pos - pI)),
                "selected_tti": float(time_to_intercept(pI, th.pos, vI, th.vel)),
                "selected_ttb": float(time_to_boundary_x0(th.pos, th.vel)),
                "selected_slack": s,
                "selected_feasible": bool(s >= -EPS),
                "selected_lead_x": float(lead[0]) if np.all(np.isfinite(lead)) else np.nan,
                "selected_lead_x_lt_0": bool(np.all(np.isfinite(lead)) and lead[0] < -EPS),
            }

    return {
        "selected_exists": False,
        "selected_dist": np.nan,
        "selected_tti": np.nan,
        "selected_ttb": np.nan,
        "selected_slack": np.nan,
        "selected_feasible": False,
        "selected_lead_x": np.nan,
        "selected_lead_x_lt_0": False,
    }


def diagnose_row(row: pd.Series, generator_seed: int) -> list[dict[str, Any]]:
    env = replay_behavior_to_selected_state(row, generator_seed=generator_seed)
    active = env.active_threats()
    pI = env.interceptor_pos.copy()
    vI = float(env.p.v_interceptor)

    heuristics = make_heuristics(seed=env.p.seed)
    actual = {h: heuristics[h](active, pI, vI) for h in HEURISTICS}
    expected = expected_decisions(active, pI, vI)

    rows = []
    for h in HEURISTICS:
        metrics = target_metrics(active, pI, vI, actual[h])
        rule_mismatch = False
        if h in {"NI", "FNI", "FMTTB", "MPS"}:
            rule_mismatch = normalize_target_id(actual[h]) != normalize_target_id(expected[h])

        rows.append(
            {
                "scenario": row.get("scenario"),
                "behavior_heuristic": row.get("behavior_heuristic"),
                "behavior_preempt": row.get("behavior_preempt"),
                "state_id": int(row.get("state_id", -1)),
                "t": float(env.t),
                "winner": row.get("winner"),
                "N_active": len(active),
                "count_feasible": int(sum(slack(pI, th, vI) >= -EPS for th in active)),
                "interceptor_x": float(pI[0]),
                "interceptor_y": float(pI[1]),
                "heuristic": h,
                "actual_choice": normalize_target_id(actual[h]),
                "expected_choice": normalize_target_id(expected[h]) if h in expected else -1,
                "rule_mismatch": bool(rule_mismatch),
                **metrics,
            }
        )
    return rows


def summarize(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_h = []
    for h, g in details.groupby("heuristic"):
        by_h.append(
            {
                "heuristic": h,
                "rows": len(g),
                "rule_mismatch_rate": float(g["rule_mismatch"].mean()),
                "selected_infeasible_rate": float((~g["selected_feasible"]).mean()),
                "selected_lead_x_lt_0_rate": float(g["selected_lead_x_lt_0"].mean()),
                "mean_selected_slack": float(g["selected_slack"].mean()),
                "median_selected_slack": float(g["selected_slack"].median()),
            }
        )

    by_h_df = pd.DataFrame(by_h).sort_values("heuristic")

    ni_rows = details[(details["heuristic"] == "NI") & (details["winner"] == "NI")]
    overall = pd.DataFrame(
        [
            {
                "checked_state_rows": int(details[["scenario", "behavior_heuristic", "state_id"]].drop_duplicates().shape[0]),
                "detail_rows": int(len(details)),
                "NI_rule_mismatches": int(details[(details["heuristic"] == "NI") & details["rule_mismatch"]].shape[0]),
                "FNI_rule_mismatches": int(details[(details["heuristic"] == "FNI") & details["rule_mismatch"]].shape[0]),
                "FMTTB_rule_mismatches": int(details[(details["heuristic"] == "FMTTB") & details["rule_mismatch"]].shape[0]),
                "MPS_rule_mismatches": int(details[(details["heuristic"] == "MPS") & details["rule_mismatch"]].shape[0]),
                "NI_winner_rows": int(len(ni_rows)),
                "NI_winner_selected_infeasible_rate": float((~ni_rows["selected_feasible"]).mean()) if len(ni_rows) else np.nan,
                "NI_winner_lead_x_lt_0_rate": float(ni_rows["selected_lead_x_lt_0"].mean()) if len(ni_rows) else np.nan,
                "NI_winner_mean_count_feasible": float(ni_rows["count_feasible"].mean()) if len(ni_rows) else np.nan,
            }
        ]
    )
    return overall, by_h_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether heuristic choices are internally consistent and "
            "whether selected targets are infeasible or have predicted intercept "
            "points beyond x=0. Useful for debugging why NI may be strong."
        )
    )
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--dataset-mode", choices=["no_ties", "with_ties", "all"], default="no_ties")
    parser.add_argument("--generator-seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "heuristic_decision_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = dataset_path(input_dir, args.dataset_mode)
    print("=== Heuristic Decision Diagnostics ===")
    print(f"Input dataset:  {path}")
    print(f"Output dir:     {output_dir}")
    print(f"Dataset mode:   {args.dataset_mode}")
    print(f"Max rows:       {args.max_rows}")

    df = pd.read_csv(path)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.random_seed).sort_index().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    all_rows = []
    for i, row in df.iterrows():
        if (i + 1) % 50 == 0 or i == 0 or i + 1 == len(df):
            print(f"Processing {i + 1}/{len(df)}")
        all_rows.extend(diagnose_row(row, generator_seed=args.generator_seed))

    details = pd.DataFrame(all_rows)
    overall, by_h = summarize(details)

    details_path = output_dir / "heuristic_decision_diagnostics_detail.csv"
    overall_path = output_dir / "heuristic_decision_diagnostics_overall.csv"
    by_h_path = output_dir / "heuristic_decision_diagnostics_by_heuristic.csv"

    details.to_csv(details_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_h.to_csv(by_h_path, index=False)

    print("\nOverall")
    print("-------")
    print(overall.to_string(index=False))

    print("\nBy heuristic")
    print("------------")
    print(by_h.to_string(index=False))

    print("\nSaved:")
    print(details_path)
    print(overall_path)
    print(by_h_path)


if __name__ == "__main__":
    main()
