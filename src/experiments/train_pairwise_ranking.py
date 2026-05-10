from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


CANDIDATE_HEURISTICS = [
    "NI",
    "MPS",
    "FNI",
    "Ratio",
    "Danger",
    "Cluster",
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

    df["failure_risk"] = (df["count_negative_slack"] + 1.0) / (df["N_active"] + 1.0)

    df["urgency_index"] = df["negative_slack_fraction"]
    df["feasible_fraction"] = 1.0 - df["negative_slack_fraction"]

    df["density_per_target"] = df["cluster_index"] / (df["N_active"] + eps)
    df["proximity_signal"] = 1.0 / (df["mean_tti"] + eps)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(1e6)

    return df


def load_rollout_data(
    path: str = "rollout_dataset_balanced_complex.csv",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_derived_features(df)

    missing = []
    for h in CANDIDATE_HEURISTICS:
        for suffix in ["future_intercepted", "future_escaped"]:
            col = f"{h}_{suffix}"
            if col not in df.columns:
                missing.append(col)

    if missing:
        raise ValueError(f"Missing rollout columns: {missing}")

    return df


def utility_for_row(row: pd.Series, heuristic: str, escaped_penalty: float = 1.0) -> float:
    return (
        row[f"{heuristic}_future_intercepted"]
        - escaped_penalty * row[f"{heuristic}_future_escaped"]
    )


def build_pairwise_dataset(
    df: pd.DataFrame,
    escaped_penalty: float = 1.0,
    skip_equal_pairs: bool = True,
) -> pd.DataFrame:
    rows = []

    for state_uid, row in df.iterrows():
        utilities = {
            h: utility_for_row(row, h, escaped_penalty=escaped_penalty)
            for h in CANDIDATE_HEURISTICS
        }

        for h_a, h_b in itertools.combinations(CANDIDATE_HEURISTICS, 2):
            u_a = utilities[h_a]
            u_b = utilities[h_b]

            if skip_equal_pairs and u_a == u_b:
                continue

            y = int(u_a > u_b)

            base = {
                "state_uid": state_uid,
                "scenario": row["scenario"],
                "seed": row.get("seed", np.nan),
                "behavior_heuristic": row.get("behavior_heuristic", ""),
                "state_id": row.get("state_id", np.nan),
                "heuristic_a": h_a,
                "heuristic_b": h_b,
                "utility_a": u_a,
                "utility_b": u_b,
                "label_a_better": y,
            }

            for col in FEATURE_COLS:
                base[col] = row[col]

            rows.append(base)

    return pd.DataFrame(rows)


def build_model() -> Pipeline:
    numeric_features = FEATURE_COLS
    categorical_features = ["heuristic_a", "heuristic_b"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", clf),
    ])


def split_by_scenario(pair_df: pd.DataFrame):
    X = pair_df[FEATURE_COLS + ["heuristic_a", "heuristic_b"]]
    y = pair_df["label_a_better"]
    groups = pair_df["scenario"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return pair_df.iloc[train_idx].copy(), pair_df.iloc[test_idx].copy()


def score_state_by_pairwise_votes(
    model: Pipeline,
    state_row: pd.Series,
) -> dict[str, float]:
    scores = {h: 0.0 for h in CANDIDATE_HEURISTICS}

    pair_rows = []

    pairs = list(itertools.combinations(CANDIDATE_HEURISTICS, 2))

    for h_a, h_b in pairs:
        row = {
            "heuristic_a": h_a,
            "heuristic_b": h_b,
        }

        for col in FEATURE_COLS:
            row[col] = state_row[col]

        pair_rows.append(row)

    X_pairs = pd.DataFrame(pair_rows)
    proba = model.predict_proba(X_pairs)[:, 1]

    for (h_a, h_b), p_a_better in zip(pairs, proba):
        scores[h_a] += p_a_better
        scores[h_b] += 1.0 - p_a_better

    return scores


def evaluate_pairwise_ranking(
    model: Pipeline,
    rollout_df: pd.DataFrame,
    output_path: str = "pairwise_ranking_predictions_by_state.csv",
) -> pd.DataFrame:
    results = []

    for state_uid, row in rollout_df.iterrows():
        true_utilities = {
            h: utility_for_row(row, h)
            for h in CANDIDATE_HEURISTICS
        }

        true_best = max(true_utilities.values())
        true_winners = {
            h for h, u in true_utilities.items()
            if u == true_best
        }

        pred_scores = score_state_by_pairwise_votes(model, row)
        pred_order = sorted(pred_scores, key=pred_scores.get, reverse=True)

        pred_top1 = pred_order[0]
        pred_top2 = set(pred_order[:2])
        pred_top3 = set(pred_order[:3])

        chosen_utility = true_utilities[pred_top1]
        regret = true_best - chosen_utility

        result = {
            "state_uid": state_uid,
            "scenario": row["scenario"],
            "behavior_heuristic": row.get("behavior_heuristic", ""),
            "N_active": row["N_active"],
            "true_best_utility": true_best,
            "true_winner_set": ",".join(sorted(true_winners)),
            "pred_top1": pred_top1,
            "pred_top2": ",".join(pred_order[:2]),
            "pred_top3": ",".join(pred_order[:3]),
            "top1_hit": pred_top1 in true_winners,
            "top2_hit": len(pred_top2.intersection(true_winners)) > 0,
            "top3_hit": len(pred_top3.intersection(true_winners)) > 0,
            "regret": regret,
        }

        for h in CANDIDATE_HEURISTICS:
            result[f"true_utility_{h}"] = true_utilities[h]
            result[f"pred_score_{h}"] = pred_scores[h]

        results.append(result)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    print("\n=== Pairwise Ranking Evaluation ===")
    print("States evaluated:", len(result_df))
    print("Top-1 hit:", round(result_df["top1_hit"].mean(), 3))
    print("Top-2 hit:", round(result_df["top2_hit"].mean(), 3))
    print("Top-3 hit:", round(result_df["top3_hit"].mean(), 3))
    print("Mean regret:", round(result_df["regret"].mean(), 3))
    print("Median regret:", round(result_df["regret"].median(), 3))
    print("Zero-regret states:", round((result_df["regret"] == 0).mean(), 3))
    print(f"Saved: {output_path}")

    return result_df


def plot_regret_distribution(
    result_df: pd.DataFrame,
    save_path: str = "pairwise_regret_distribution.png",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(result_df["regret"], bins=30)
    plt.title("Pairwise Ranking Regret Distribution")
    plt.xlabel("Regret")
    plt.ylabel("Number of states")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")


def plot_top1_distribution(
    result_df: pd.DataFrame,
    save_path: str = "pairwise_predicted_top1_distribution.png",
) -> None:
    counts = result_df["pred_top1"].value_counts().sort_values()

    plt.figure(figsize=(8, 5))
    plt.barh(counts.index, counts.values)
    plt.title("Pairwise Ranking Predicted Top-1 Distribution")
    plt.xlabel("Number of states")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")


def main() -> None:
    input_path = "rollout_dataset_balanced_complex.csv"

    rollout_df = load_rollout_data(input_path)

    print("\n=== Loaded rollout dataset ===")
    print(rollout_df.shape)

    pair_df = build_pairwise_dataset(
        rollout_df,
        escaped_penalty=1.0,
        skip_equal_pairs=True,
    )

    pair_df.to_csv("pairwise_training_dataset.csv", index=False)

    print("\n=== Pairwise dataset ===")
    print(pair_df.shape)
    print("\nLabel distribution:")
    print(pair_df["label_a_better"].value_counts(normalize=True).round(3))

    train_df, test_pair_df = split_by_scenario(pair_df)

    test_scenarios = set(test_pair_df["scenario"].unique())
    test_rollout_df = rollout_df[rollout_df["scenario"].isin(test_scenarios)].copy()

    print("\n=== Scenario split ===")
    print("Train pair rows:", len(train_df))
    print("Test pair rows:", len(test_pair_df))
    print("Test rollout states:", len(test_rollout_df))
    print("Train scenarios:", train_df["scenario"].nunique())
    print("Test scenarios:", test_pair_df["scenario"].nunique())

    model = build_model()

    model.fit(
        train_df[FEATURE_COLS + ["heuristic_a", "heuristic_b"]],
        train_df["label_a_better"],
    )

    y_test = test_pair_df["label_a_better"]
    y_pred = model.predict(test_pair_df[FEATURE_COLS + ["heuristic_a", "heuristic_b"]])

    print("\n=== Pairwise classification accuracy ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

    result_df = evaluate_pairwise_ranking(
        model,
        test_rollout_df,
        output_path="pairwise_ranking_predictions_by_state.csv",
    )

    plot_regret_distribution(result_df)
    plot_top1_distribution(result_df)

    print("\nSaved:")
    print("- pairwise_training_dataset.csv")
    print("- pairwise_ranking_predictions_by_state.csv")
    print("- pairwise_regret_distribution.png")
    print("- pairwise_predicted_top1_distribution.png")


if __name__ == "__main__":
    main()
