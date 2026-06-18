from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


CANDIDATE_HEURISTICS = [
    "NI",
    "FNI",
    "FMTTB",
    "MPS",
    "FCluster",
]

BASE_FEATURE_COLS = [
    "N_active",
    "min_ttb",
    "mean_ttb",
    "min_positive_slack",
    "mean_slack",
    "count_negative_slack",
    "mean_tti",
    "cluster_index",
]

DERIVED_FEATURE_COLS = [
    "mean_tti_to_mean_ttb_ratio",
    "mean_slack_to_mean_ttb_ratio",
    "negative_slack_fraction",
    "min_ttb_to_mean_ttb_ratio",
    "ttb_spread",
    "ttb_relative_spread",
    "slack_pressure",
    "min_vs_mean_slack",
    "failure_risk",
    "urgency_index",
    "feasible_fraction",
    "density_per_target",
    "proximity_signal",
]

FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS

STATE_ID_COLS = [
    "scenario",
    "seed",
    "behavior_heuristic",
    "state_id",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    df = df.replace([np.inf, -np.inf], np.nan)

    df["mean_tti_to_mean_ttb_ratio"] = df["mean_tti"] / (df["mean_ttb"] + eps)
    df["mean_slack_to_mean_ttb_ratio"] = df["mean_slack"] / (df["mean_ttb"] + eps)
    df["negative_slack_fraction"] = df["count_negative_slack"] / (df["N_active"] + eps)

    df["min_ttb_to_mean_ttb_ratio"] = df["min_ttb"] / (df["mean_ttb"] + eps)
    df["ttb_spread"] = df["mean_ttb"] - df["min_ttb"]
    df["ttb_relative_spread"] = df["ttb_spread"] / (df["mean_ttb"] + eps)

    df["slack_pressure"] = -df["mean_slack"]
    df["min_vs_mean_slack"] = df["min_positive_slack"] - df["mean_slack"]

    df["failure_risk"] = (
        df["count_negative_slack"] + 1.0
    ) / (df["N_active"] + 1.0)

    df["urgency_index"] = df["negative_slack_fraction"]
    df["feasible_fraction"] = 1.0 - df["negative_slack_fraction"]

    df["density_per_target"] = df["cluster_index"] / (df["N_active"] + eps)
    df["proximity_signal"] = 1.0 / (df["mean_tti"] + eps)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(1e6)

    return df


def load_rollout_data(
    path: str = "rollout_dataset_compact_heuristics_complex.csv",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_derived_features(df)

    missing = []
    for h in CANDIDATE_HEURISTICS:
        if f"{h}_future_intercepted" not in df.columns:
            missing.append(f"{h}_future_intercepted")
        if f"{h}_future_escaped" not in df.columns:
            missing.append(f"{h}_future_escaped")

    if missing:
        raise ValueError(f"Missing rollout columns: {missing}")

    return df


def build_long_ranking_dataset(
    df: pd.DataFrame,
    escaped_penalty: float = 1.0,
) -> pd.DataFrame:
    """
    Converts one row per state into one row per (state, heuristic).

    Target score:
        utility = future_intercepted - escaped_penalty * future_escaped

    This avoids forcing a single label. The model learns how good each
    heuristic is under each state.
    """

    rows = []

    for idx, row in df.iterrows():
        state_uid = idx

        base = {
            "state_uid": state_uid,
        }

        for col in STATE_ID_COLS:
            if col in df.columns:
                base[col] = row[col]

        for col in FEATURE_COLS:
            base[col] = row[col]

        for h in CANDIDATE_HEURISTICS:
            future_intercepted = row[f"{h}_future_intercepted"]
            future_escaped = row[f"{h}_future_escaped"]
            utility = future_intercepted - escaped_penalty * future_escaped

            rows.append({
                **base,
                "heuristic": h,
                "future_intercepted": future_intercepted,
                "future_escaped": future_escaped,
                "utility": utility,
            })

    long_df = pd.DataFrame(rows)
    return long_df


def build_model() -> Pipeline:
    numeric_features = FEATURE_COLS
    categorical_features = ["heuristic"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def split_by_scenario(long_df: pd.DataFrame):
    """
    Split by scenario, not random rows.

    This prevents leakage from the same scenario into both train and test.
    """

    X = long_df[FEATURE_COLS + ["heuristic"]]
    y = long_df["utility"]
    groups = long_df["scenario"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_df = long_df.iloc[train_idx].copy()
    test_df = long_df.iloc[test_idx].copy()

    return train_df, test_df


def evaluate_ranking(
    test_df: pd.DataFrame,
    pred_col: str = "pred_utility",
    true_col: str = "utility",
    output_path: str = "ranking_predictions_by_state.csv",
) -> pd.DataFrame:
    """
    Evaluate ranking per state.

    Metrics:
    - Top-1 hit: predicted best heuristic is truly optimal.
    - Top-2 hit: at least one of predicted top-2 is truly optimal.
    - Top-3 hit.
    - Regret: true best utility - true utility of predicted top-1.
    """

    state_results = []

    for state_uid, g in test_df.groupby("state_uid"):
        g = g.copy()

        true_max = g[true_col].max()
        true_winners = set(g[g[true_col] == true_max]["heuristic"])

        ranked = g.sort_values(pred_col, ascending=False)
        pred_order = ranked["heuristic"].tolist()

        top1 = pred_order[0]
        top2 = set(pred_order[:2])
        top3 = set(pred_order[:3])

        true_top1_utility = float(g[g["heuristic"] == top1][true_col].iloc[0])
        regret = float(true_max - true_top1_utility)

        row = {
            "state_uid": state_uid,
            "scenario": g["scenario"].iloc[0],
            "behavior_heuristic": g["behavior_heuristic"].iloc[0],
            "t": g["t"].iloc[0] if "t" in g.columns else np.nan,
            "N_active": g["N_active"].iloc[0],
            "true_best_utility": true_max,
            "true_winner_set": ",".join(sorted(true_winners)),
            "pred_top1": top1,
            "pred_top2": ",".join(pred_order[:2]),
            "pred_top3": ",".join(pred_order[:3]),
            "top1_hit": top1 in true_winners,
            "top2_hit": len(top2.intersection(true_winners)) > 0,
            "top3_hit": len(top3.intersection(true_winners)) > 0,
            "regret": regret,
        }

        for _, r in ranked.iterrows():
            h = r["heuristic"]
            row[f"pred_score_{h}"] = r[pred_col]
            row[f"true_utility_{h}"] = r[true_col]

        state_results.append(row)

    result_df = pd.DataFrame(state_results)

    result_df.to_csv(output_path, index=False)

    print("\n=== Ranking Evaluation ===")
    print("States evaluated:", len(result_df))
    print("Top-1 hit:", round(result_df["top1_hit"].mean(), 3))
    print("Top-2 hit:", round(result_df["top2_hit"].mean(), 3))
    print("Top-3 hit:", round(result_df["top3_hit"].mean(), 3))
    print("Mean regret:", round(result_df["regret"].mean(), 3))
    print("Median regret:", round(result_df["regret"].median(), 3))
    print("Zero-regret states:", round((result_df["regret"] == 0).mean(), 3))

    print(f"\nSaved ranking predictions: {output_path}")

    return result_df


def scenario_level_cv(long_df: pd.DataFrame, n_splits: int = 5) -> None:
    """
    Simple scenario-level CV for ranking.

    Trains a regressor and evaluates Top-1/Top-2/Regret by held-out scenarios.
    """

    groups = long_df["scenario"].unique()
    n_splits = min(n_splits, len(groups))

    gkf = GroupKFold(n_splits=n_splits)

    X = long_df[FEATURE_COLS + ["heuristic"]]
    y = long_df["utility"]
    group_labels = long_df["scenario"]

    top1_scores = []
    top2_scores = []
    regrets = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=group_labels),
        start=1,
    ):
        train_df = long_df.iloc[train_idx].copy()
        test_df = long_df.iloc[test_idx].copy()

        model = build_model()

        model.fit(
            train_df[FEATURE_COLS + ["heuristic"]],
            train_df["utility"],
        )

        test_df["pred_utility"] = model.predict(
            test_df[FEATURE_COLS + ["heuristic"]]
        )

        result_df = evaluate_ranking(
            test_df,
            output_path=f"ranking_predictions_fold_{fold}.csv",
        )

        top1_scores.append(result_df["top1_hit"].mean())
        top2_scores.append(result_df["top2_hit"].mean())
        regrets.append(result_df["regret"].mean())

    print("\n=== Scenario-level Ranking Cross Validation ===")
    print("Top-1 scores:", np.round(top1_scores, 3))
    print("Top-1 mean:", round(float(np.mean(top1_scores)), 3))
    print("Top-2 scores:", np.round(top2_scores, 3))
    print("Top-2 mean:", round(float(np.mean(top2_scores)), 3))
    print("Mean regret per fold:", np.round(regrets, 3))
    print("Mean regret:", round(float(np.mean(regrets)), 3))


def plot_regret_distribution(
    result_df: pd.DataFrame,
    save_path: str = "ranking_regret_distribution.png",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(result_df["regret"], bins=30)
    plt.title("Ranking Model Regret Distribution")
    plt.xlabel("Regret = best true utility - chosen heuristic utility")
    plt.ylabel("Number of states")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nSaved regret distribution plot: {save_path}")


def plot_predicted_top1_distribution(
    result_df: pd.DataFrame,
    save_path: str = "ranking_predicted_top1_distribution.png",
) -> None:
    counts = result_df["pred_top1"].value_counts().sort_values()

    plt.figure(figsize=(8, 5))
    plt.barh(counts.index, counts.values)
    plt.title("Predicted Top-1 Heuristic Distribution")
    plt.xlabel("Number of states")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nSaved predicted Top-1 distribution plot: {save_path}")


def main() -> None:
    input_path = "rollout_dataset_compact_heuristics_complex.csv"

    df = load_rollout_data(input_path)

    print("\n=== Loaded rollout dataset ===")
    print(df.shape)

    long_df = build_long_ranking_dataset(
        df,
        escaped_penalty=1.0,
    )

    long_df.to_csv("ranking_dataset_long.csv", index=False)

    print("\n=== Long ranking dataset ===")
    print(long_df.shape)
    print("\nRows per heuristic:")
    print(long_df["heuristic"].value_counts())

    train_df, test_df = split_by_scenario(long_df)

    print("\n=== Scenario-level train/test split ===")
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Train states:", train_df["state_uid"].nunique())
    print("Test states:", test_df["state_uid"].nunique())
    print("Train scenarios:", train_df["scenario"].nunique())
    print("Test scenarios:", test_df["scenario"].nunique())

    model = build_model()

    model.fit(
        train_df[FEATURE_COLS + ["heuristic"]],
        train_df["utility"],
    )

    test_df = test_df.copy()
    test_df["pred_utility"] = model.predict(
        test_df[FEATURE_COLS + ["heuristic"]]
    )

    print("\n=== Regression quality on heuristic-state pairs ===")
    y_true = test_df["utility"]
    y_pred = test_df["pred_utility"]
    print("MAE:", round(mean_absolute_error(y_true, y_pred), 3))
    print("RMSE:", round(mean_squared_error(y_true, y_pred) ** 0.5, 3))

    result_df = evaluate_ranking(
        test_df,
        output_path="ranking_predictions_by_state.csv",
    )

    plot_regret_distribution(
        result_df,
        save_path="ranking_regret_distribution.png",
    )

    plot_predicted_top1_distribution(
        result_df,
        save_path="ranking_predicted_top1_distribution.png",
    )

    scenario_level_cv(long_df, n_splits=5)

    print("\nSaved:")
    print("- ranking_dataset_long.csv")
    print("- ranking_predictions_by_state.csv")
    print("- ranking_regret_distribution.png")
    print("- ranking_predicted_top1_distribution.png")


if __name__ == "__main__":
    main()
