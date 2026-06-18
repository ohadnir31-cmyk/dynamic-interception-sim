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
    if dataset_mode == "no_ties":
        filename = "large_scale_rollout_states_informative_no_ties.csv"
    elif dataset_mode == "with_ties":
        filename = "large_scale_rollout_states_informative_with_ties.csv"
    else:
        filename = "large_scale_rollout_states.csv"

    path = input_dir / filename
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
        if all(col in df.columns for col in required):
            found.append(h)
    if not found:
        raise ValueError(
            "No heuristic outcome columns found. Expected columns such as "
            "NI_future_intercepted, NI_regret, NI_rank."
        )
    return found


def validate_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing observable feature columns: {missing}")
    return features


def add_active_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "N_active_bucket" in df.columns:
        return df
    bins = [-np.inf, 1, 3, 6, 10, np.inf]
    labels = ["1", "2-3", "4-6", "7-10", "11+"]
    df = df.copy()
    df["N_active_bucket"] = pd.cut(df["N_active"], bins=bins, labels=labels)
    return df


def split_by_scenario(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def compute_winner_margin(df: pd.DataFrame, heuristics: list[str]) -> pd.Series:
    score_cols = [f"{h}_future_intercepted" for h in heuristics]
    scores = df[score_cols].to_numpy(dtype=float)
    sorted_scores = np.sort(scores, axis=1)
    best = sorted_scores[:, -1]
    second_best = sorted_scores[:, -2] if scores.shape[1] >= 2 else sorted_scores[:, -1]
    return pd.Series(best - second_best, index=df.index)


def make_sample_weights(
    train_df: pd.DataFrame,
    heuristics: list[str],
    mode: str,
    alpha: float,
) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "margin":
        margin = compute_winner_margin(train_df, heuristics)
        return (1.0 + alpha * margin).to_numpy(dtype=float)
    if mode == "oracle_gap":
        regret_cols = [f"{h}_regret" for h in heuristics]
        max_regret = train_df[regret_cols].max(axis=1)
        return (1.0 + alpha * max_regret).to_numpy(dtype=float)
    raise ValueError(f"Unknown sample-weight mode: {mode}")


def train_regret_models(
    train_df: pd.DataFrame,
    features: list[str],
    heuristics: list[str],
    random_state: int,
    sample_weights: np.ndarray | None,
) -> dict[str, RandomForestRegressor]:
    models: dict[str, RandomForestRegressor] = {}
    X_train = train_df[features].to_numpy(dtype=float)

    for h in heuristics:
        y_train = train_df[f"{h}_regret"].to_numpy(dtype=float)
        model = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        models[h] = model
    return models


def predict_regrets(
    models: dict[str, RandomForestRegressor],
    test_df: pd.DataFrame,
    features: list[str],
    heuristics: list[str],
) -> pd.DataFrame:
    X_test = test_df[features].to_numpy(dtype=float)
    pred = pd.DataFrame(index=test_df.index)

    for h in heuristics:
        pred[f"{h}_predicted_regret"] = models[h].predict(X_test)

    pred["predicted_heuristic"] = (
        pred[[f"{h}_predicted_regret" for h in heuristics]]
        .idxmin(axis=1)
        .str.replace("_predicted_regret", "", regex=False)
    )
    return pred


def add_model_outcomes(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    heuristics: list[str],
) -> pd.DataFrame:
    out = test_df.copy()
    out = pd.concat([out, pred_df], axis=1)

    model_future_intercepted = []
    model_regret = []
    model_rank = []
    model_future_escaped = []

    for _, row in out.iterrows():
        h = row["predicted_heuristic"]
        model_future_intercepted.append(row[f"{h}_future_intercepted"])
        model_regret.append(row[f"{h}_regret"])
        model_rank.append(row[f"{h}_rank"])

        escaped_col = f"{h}_future_escaped"
        if escaped_col in out.columns:
            model_future_escaped.append(row[escaped_col])

    out["model_future_intercepted"] = model_future_intercepted
    out["model_regret"] = model_regret
    out["model_rank"] = model_rank

    if model_future_escaped:
        out["model_future_escaped"] = model_future_escaped

    if "best_future_intercepted" not in out.columns:
        out["best_future_intercepted"] = out[
            [f"{h}_future_intercepted" for h in heuristics]
        ].max(axis=1)

    if "best_future_escaped" not in out.columns:
        escaped_cols = [
            f"{h}_future_escaped"
            for h in heuristics
            if f"{h}_future_escaped" in out.columns
        ]
        if escaped_cols:
            best_escaped = []
            for _, row in out.iterrows():
                best_h = max(heuristics, key=lambda hh: row[f"{hh}_future_intercepted"])
                col = f"{best_h}_future_escaped"
                best_escaped.append(row[col] if col in out.columns else np.nan)
            out["best_future_escaped"] = best_escaped

    out["zero_regret"] = out["model_regret"] == 0
    out["top1"] = out["model_rank"] <= 1
    out["top2"] = out["model_rank"] <= 2
    out["top3"] = out["model_rank"] <= 3
    return out


def summarize_selector(test_pred: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "test_rows": len(test_pred),
        "accuracy": float((test_pred["predicted_heuristic"] == test_pred["winner"]).mean())
        if "winner" in test_pred.columns
        else np.nan,
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
        summary["mean_oracle_future_escaped"] = float(test_pred["best_future_escaped"].mean())
    if "model_future_escaped" in test_pred.columns:
        summary["mean_model_future_escaped"] = float(test_pred["model_future_escaped"].mean())
    return pd.DataFrame([summary])


def summarize_by_bucket(test_pred: pd.DataFrame) -> pd.DataFrame:
    if "N_active_bucket" not in test_pred.columns:
        return pd.DataFrame()

    grouped = (
        test_pred
        .groupby("N_active_bucket", observed=True)
        .agg(
            num_rows=("model_regret", "size"),
            mean_model_regret=("model_regret", "mean"),
            median_model_regret=("model_regret", "median"),
            zero_regret_rate=("zero_regret", "mean"),
            top1_rate=("top1", "mean"),
            top2_rate=("top2", "mean"),
            top3_rate=("top3", "mean"),
            mean_oracle_future_intercepted=("best_future_intercepted", "mean"),
            mean_model_future_intercepted=("model_future_intercepted", "mean"),
        )
        .reset_index()
    )

    if "winner" in test_pred.columns:
        acc = (
            test_pred
            .groupby("N_active_bucket", observed=True)
            .apply(lambda g: (g["predicted_heuristic"] == g["winner"]).mean(), include_groups=False)
            .rename("accuracy")
            .reset_index()
        )
        grouped = grouped.merge(acc, on="N_active_bucket", how="left")

    return grouped


def summarize_model_fit(
    models: dict[str, RandomForestRegressor],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    heuristics: list[str],
) -> pd.DataFrame:
    rows = []
    X_train = train_df[features].to_numpy(dtype=float)
    X_test = test_df[features].to_numpy(dtype=float)

    for h in heuristics:
        model = models[h]
        y_train = train_df[f"{h}_regret"].to_numpy(dtype=float)
        y_test = test_df[f"{h}_regret"].to_numpy(dtype=float)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        rows.append(
            {
                "heuristic": h,
                "train_mae": mean_absolute_error(y_train, pred_train),
                "test_mae": mean_absolute_error(y_test, pred_test),
                "train_rmse": mean_squared_error(y_train, pred_train) ** 0.5,
                "test_rmse": mean_squared_error(y_test, pred_test) ** 0.5,
            }
        )
    return pd.DataFrame(rows)


def plot_regret_distribution(test_pred: pd.DataFrame, output_dir: Path) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an observable-only regret-based selector. The selector trains one "
            "RandomForestRegressor per heuristic to predict regret, then chooses the "
            "heuristic with minimum predicted regret."
        )
    )
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--dataset-mode",
        choices=["no_ties", "with_ties", "full"],
        default="no_ties",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "margin", "oracle_gap"],
        default="margin",
        help="margin emphasizes states where the best heuristic beats the runner-up by more.",
    )
    parser.add_argument("--weight-alpha", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
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

    train_df, test_df = split_by_scenario(
        df=df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Rows loaded:      {len(df)}")
    print(f"Train scenarios:  {train_df['scenario'].nunique()}")
    print(f"Test scenarios:   {test_df['scenario'].nunique()}")
    print(f"Train rows:       {len(train_df)}")
    print(f"Test rows:        {len(test_df)}")
    print(f"Heuristics:       {heuristics}")
    print(f"Features used:    {features}")

    sample_weights = make_sample_weights(
        train_df=train_df,
        heuristics=heuristics,
        mode=args.sample_weight_mode,
        alpha=args.weight_alpha,
    )

    models = train_regret_models(
        train_df=train_df,
        features=features,
        heuristics=heuristics,
        random_state=args.random_state,
        sample_weights=sample_weights,
    )

    pred_df = predict_regrets(
        models=models,
        test_df=test_df,
        features=features,
        heuristics=heuristics,
    )

    test_pred = add_model_outcomes(
        test_df=test_df,
        pred_df=pred_df,
        heuristics=heuristics,
    )

    summary = summarize_selector(test_pred)
    bucket_summary = summarize_by_bucket(test_pred)
    model_fit = summarize_model_fit(
        models=models,
        train_df=train_df,
        test_df=test_df,
        features=features,
        heuristics=heuristics,
    )

    # Save using this filename so compare_selector_to_fixed_baselines.py can be reused.
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
    print(
        "python -m src.experiments.compare_selector_to_fixed_baselines "
        f"--selector-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
