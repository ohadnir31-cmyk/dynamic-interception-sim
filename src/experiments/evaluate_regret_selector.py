from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


HEURISTICS = ["NI", "FNI", "MPS", "Danger", "Ratio", "Cluster"]

OBSERVABLE_FEATURES = [
    "N_active",
    "min_ttb",
    "mean_ttb",
    "std_ttb",
    "min_tti",
    "mean_tti",
    "std_tti",
    "min_slack",
    "mean_slack",
    "std_slack",
    "min_positive_slack",
    "count_feasible",
    "count_negative_slack",
    "feasible_ratio",
    "cluster_index",
    "spatial_spread_x",
    "spatial_spread_y",
    "spatial_dispersion",
]


def load_dataset(input_dir: Path, dataset_mode: str) -> pd.DataFrame:
    filenames = {
        "no_ties": "large_scale_rollout_states_informative_no_ties.csv",
        "with_ties": "large_scale_rollout_states_informative_with_ties.csv",
        "full": "large_scale_rollout_states.csv",
    }
    path = input_dir / filenames[dataset_mode]
    if not path.exists():
        raise FileNotFoundError(f"Could not find input dataset: {path}")
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input dataset is empty: {path}")
    return df


def infer_heuristics(df: pd.DataFrame) -> list[str]:
    found = []
    for h in HEURISTICS:
        required = [f"{h}_future_intercepted", f"{h}_regret", f"{h}_rank"]
        if all(c in df.columns for c in required):
            found.append(h)
    if not found:
        raise ValueError("No heuristic outcome columns were found.")
    return found


def validate_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing observable feature columns: {missing}")
    return features


def add_active_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "N_active_bucket" in df.columns:
        return df
    df = df.copy()
    df["N_active_bucket"] = pd.cut(
        df["N_active"],
        bins=[-np.inf, 1, 3, 6, 10, np.inf],
        labels=["1", "2-3", "4-6", "7-10", "11+"],
    )
    return df


def split_by_scenario(df: pd.DataFrame, test_size: float, random_state: int):
    if "scenario" not in df.columns:
        raise ValueError("Missing required column: scenario")
    scenarios = np.array(sorted(df["scenario"].unique()))
    train_scenarios, test_scenarios = train_test_split(
        scenarios,
        test_size=test_size,
        random_state=random_state,
    )
    train_df = df[df["scenario"].isin(train_scenarios)].copy()
    test_df = df[df["scenario"].isin(test_scenarios)].copy()
    return train_df, test_df


def clean_feature_values(train_df, test_df, features, clip_abs: float):
    """
    Replace inf/-inf/NaN in observable features using train-set medians only.
    This avoids leakage from the test set.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    rows = []

    for f in features:
        train_raw = pd.to_numeric(train_df[f], errors="coerce")
        test_raw = pd.to_numeric(test_df[f], errors="coerce")

        train_arr = train_raw.to_numpy(dtype=float)
        test_arr = test_raw.to_numpy(dtype=float)
        train_bad = int((~np.isfinite(train_arr)).sum())
        test_bad = int((~np.isfinite(test_arr)).sum())

        train_clean = train_raw.replace([np.inf, -np.inf], np.nan)
        test_clean = test_raw.replace([np.inf, -np.inf], np.nan)

        median = train_clean.median()
        if not np.isfinite(median):
            median = 0.0

        train_clean = train_clean.fillna(median)
        test_clean = test_clean.fillna(median)

        if clip_abs and clip_abs > 0:
            train_clean = train_clean.clip(-clip_abs, clip_abs)
            test_clean = test_clean.clip(-clip_abs, clip_abs)

        train_df[f] = train_clean.astype(float)
        test_df[f] = test_clean.astype(float)

        rows.append({
            "feature": f,
            "train_non_finite_or_nan_count": train_bad,
            "test_non_finite_or_nan_count": test_bad,
            "median_imputation_value": float(median),
            "train_min_after_cleaning": float(train_df[f].min()),
            "train_max_after_cleaning": float(train_df[f].max()),
            "test_min_after_cleaning": float(test_df[f].min()),
            "test_max_after_cleaning": float(test_df[f].max()),
        })

    if not np.isfinite(train_df[features].to_numpy(dtype=float)).all():
        raise ValueError("Training feature matrix still contains non-finite values after cleaning.")
    if not np.isfinite(test_df[features].to_numpy(dtype=float)).all():
        raise ValueError("Test feature matrix still contains non-finite values after cleaning.")

    return train_df, test_df, pd.DataFrame(rows)


def compute_winner_margin(df: pd.DataFrame, heuristics: list[str]) -> pd.Series:
    scores = df[[f"{h}_future_intercepted" for h in heuristics]].to_numpy(dtype=float)
    sorted_scores = np.sort(scores, axis=1)
    return pd.Series(sorted_scores[:, -1] - sorted_scores[:, -2], index=df.index)


def make_sample_weights(train_df, heuristics, mode: str, alpha: float):
    if mode == "none":
        return None
    if mode == "margin":
        margin = compute_winner_margin(train_df, heuristics)
        return (1.0 + alpha * margin).to_numpy(dtype=float)
    if mode == "oracle_gap":
        max_regret = train_df[[f"{h}_regret" for h in heuristics]].max(axis=1)
        return (1.0 + alpha * max_regret).to_numpy(dtype=float)
    raise ValueError(f"Unknown sample-weight mode: {mode}")


def train_regret_models(train_df, features, heuristics, random_state, sample_weights):
    X = train_df[features].to_numpy(dtype=float)
    models = {}
    for h in heuristics:
        y = train_df[f"{h}_regret"].to_numpy(dtype=float)
        model = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X, y, sample_weight=sample_weights)
        models[h] = model
    return models


def predict_regrets(models, test_df, features, heuristics):
    X = test_df[features].to_numpy(dtype=float)
    pred = pd.DataFrame(index=test_df.index)
    for h in heuristics:
        pred[f"{h}_predicted_regret"] = models[h].predict(X)
    pred["predicted_heuristic"] = (
        pred[[f"{h}_predicted_regret" for h in heuristics]]
        .idxmin(axis=1)
        .str.replace("_predicted_regret", "", regex=False)
    )
    return pred


def add_model_outcomes(test_df, pred_df, heuristics):
    out = pd.concat([test_df.copy(), pred_df], axis=1)

    model_future_intercepted = []
    model_regret = []
    model_rank = []
    model_future_escaped = []
    has_escaped = any(f"{h}_future_escaped" in out.columns for h in heuristics)

    for _, row in out.iterrows():
        h = row["predicted_heuristic"]
        model_future_intercepted.append(row[f"{h}_future_intercepted"])
        model_regret.append(row[f"{h}_regret"])
        model_rank.append(row[f"{h}_rank"])
        if has_escaped:
            col = f"{h}_future_escaped"
            model_future_escaped.append(row[col] if col in out.columns else np.nan)

    out["model_future_intercepted"] = model_future_intercepted
    out["model_regret"] = model_regret
    out["model_rank"] = model_rank
    if has_escaped:
        out["model_future_escaped"] = model_future_escaped

    if "best_future_intercepted" not in out.columns:
        out["best_future_intercepted"] = out[[f"{h}_future_intercepted" for h in heuristics]].max(axis=1)

    if "winner" not in out.columns:
        out["winner"] = out[[f"{h}_future_intercepted" for h in heuristics]].idxmax(axis=1).str.replace("_future_intercepted", "", regex=False)

    if "best_future_escaped" not in out.columns and has_escaped:
        vals = []
        for _, row in out.iterrows():
            best_h = max(heuristics, key=lambda hh: row[f"{hh}_future_intercepted"])
            vals.append(row.get(f"{best_h}_future_escaped", np.nan))
        out["best_future_escaped"] = vals

    out["zero_regret"] = out["model_regret"] == 0
    out["top1"] = out["model_rank"] <= 1
    out["top2"] = out["model_rank"] <= 2
    out["top3"] = out["model_rank"] <= 3
    return out


def summarize_selector(test_pred: pd.DataFrame) -> pd.DataFrame:
    row = {
        "test_rows": len(test_pred),
        "accuracy": float((test_pred["predicted_heuristic"] == test_pred["winner"]).mean()),
        "mean_model_regret": float(test_pred["model_regret"].mean()),
        "median_model_regret": float(test_pred["model_regret"].median()),
        "max_model_regret": float(test_pred["model_regret"].max()),
        "zero_regret_rate": float(test_pred["zero_regret"].mean()),
        "top1_rate": float(test_pred["top1"].mean()),
        "top2_rate": float(test_pred["top2"].mean()),
        "top3_rate": float(test_pred["top3"].mean()),
        "mean_oracle_future_intercepted": float(test_pred["best_future_intercepted"].mean()),
        "mean_model_future_intercepted": float(test_pred["model_future_intercepted"].mean()),
    }
    if "best_future_escaped" in test_pred.columns:
        row["mean_oracle_future_escaped"] = float(test_pred["best_future_escaped"].mean())
    if "model_future_escaped" in test_pred.columns:
        row["mean_model_future_escaped"] = float(test_pred["model_future_escaped"].mean())
    return pd.DataFrame([row])


def summarize_by_bucket(test_pred: pd.DataFrame) -> pd.DataFrame:
    if "N_active_bucket" not in test_pred.columns:
        return pd.DataFrame()
    rows = []
    for bucket, group in test_pred.groupby("N_active_bucket", observed=True):
        rows.append({
            "N_active_bucket": bucket,
            "num_rows": len(group),
            "accuracy": float((group["predicted_heuristic"] == group["winner"]).mean()),
            "mean_model_regret": float(group["model_regret"].mean()),
            "median_model_regret": float(group["model_regret"].median()),
            "zero_regret_rate": float(group["zero_regret"].mean()),
            "top1_rate": float(group["top1"].mean()),
            "top2_rate": float(group["top2"].mean()),
            "top3_rate": float(group["top3"].mean()),
            "mean_oracle_future_intercepted": float(group["best_future_intercepted"].mean()),
            "mean_model_future_intercepted": float(group["model_future_intercepted"].mean()),
        })
    return pd.DataFrame(rows)


def summarize_model_fit(models, train_df, test_df, features, heuristics):
    X_train = train_df[features].to_numpy(dtype=float)
    X_test = test_df[features].to_numpy(dtype=float)
    rows = []
    for h in heuristics:
        y_train = train_df[f"{h}_regret"].to_numpy(dtype=float)
        y_test = test_df[f"{h}_regret"].to_numpy(dtype=float)
        pred_train = models[h].predict(X_train)
        pred_test = models[h].predict(X_test)
        rows.append({
            "heuristic": h,
            "train_mae": mean_absolute_error(y_train, pred_train),
            "test_mae": mean_absolute_error(y_test, pred_test),
            "train_rmse": mean_squared_error(y_train, pred_train) ** 0.5,
            "test_rmse": mean_squared_error(y_test, pred_test) ** 0.5,
        })
    return pd.DataFrame(rows)


def plot_regret_distribution(test_pred, output_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(test_pred["model_regret"], bins=30)
    plt.xlabel("Model regret")
    plt.ylabel("Number of states")
    plt.title("Regret-selector regret distribution")
    plt.tight_layout()
    path = output_dir / "regret_selector_regret_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an observable-only regret-based selector.")
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--dataset-mode", choices=["no_ties", "with_ties", "full"], default="no_ties")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-weight-mode", choices=["none", "margin", "oracle_gap"], default="margin")
    parser.add_argument("--weight-alpha", type=float, default=0.25)
    parser.add_argument("--clip-abs", type=float, default=1_000_000.0)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Observable-Only Regret Selector ===")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset mode:     {args.dataset_mode}")
    print(f"Sample weights:   {args.sample_weight_mode}, alpha={args.weight_alpha}")

    df = load_dataset(input_dir, args.dataset_mode)
    df = add_active_bucket(df)
    heuristics = infer_heuristics(df)
    features = validate_features(df, OBSERVABLE_FEATURES)

    train_df, test_df = split_by_scenario(df, args.test_size, args.random_state)
    train_df, test_df, cleaning_report = clean_feature_values(train_df, test_df, features, args.clip_abs)
    cleaning_report.to_csv(output_dir / "feature_cleaning_report.csv", index=False)

    bad_total = int(
        cleaning_report["train_non_finite_or_nan_count"].sum()
        + cleaning_report["test_non_finite_or_nan_count"].sum()
    )

    print(f"Rows loaded:      {len(df)}")
    print(f"Train scenarios:  {train_df['scenario'].nunique()}")
    print(f"Test scenarios:   {test_df['scenario'].nunique()}")
    print(f"Train rows:       {len(train_df)}")
    print(f"Test rows:        {len(test_df)}")
    print(f"Heuristics:       {heuristics}")
    print(f"Features used:    {features}")
    print(f"Non-finite/NaN feature values cleaned: {bad_total}")
    print(f"Cleaning report:  {output_dir / 'feature_cleaning_report.csv'}")

    sample_weights = make_sample_weights(train_df, heuristics, args.sample_weight_mode, args.weight_alpha)
    models = train_regret_models(train_df, features, heuristics, args.random_state, sample_weights)
    pred_df = predict_regrets(models, test_df, features, heuristics)
    test_pred = add_model_outcomes(test_df, pred_df, heuristics)

    summary = summarize_selector(test_pred)
    bucket_summary = summarize_by_bucket(test_pred)
    model_fit = summarize_model_fit(models, train_df, test_df, features, heuristics)

    test_pred.to_csv(output_dir / "selector_evaluation_test_predictions.csv", index=False)
    test_pred.to_csv(output_dir / "regret_selector_test_predictions.csv", index=False)
    summary.to_csv(output_dir / "regret_selector_summary.csv", index=False)
    bucket_summary.to_csv(output_dir / "regret_selector_by_active_bucket.csv", index=False)
    model_fit.to_csv(output_dir / "regret_regressor_fit_by_heuristic.csv", index=False)

    plot_regret_distribution(test_pred, output_dir)

    print("\nSummary")
    print("-------")
    print(summary.to_string(index=False))
    print("\nBy active-target bucket")
    print("-----------------------")
    print(bucket_summary.to_string(index=False))
    print("\nRegret-regressor fit by heuristic")
    print("---------------------------------")
    print(model_fit.to_string(index=False))
    print("\nDone.")
    print(f"Saved outputs to: {output_dir}")
    print("\nNext recommended command:")
    print("python -m src.experiments.compare_selector_to_fixed_baselines " f"--selector-dir {output_dir}")


if __name__ == "__main__":
    main()
