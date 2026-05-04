from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.heuristic_groups import HEURISTIC_TO_GROUP


# ============================================================
# Feature definitions
# ============================================================

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


# ============================================================
# Data loading and feature engineering
# ============================================================

def add_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["winner_group"] = df["winner"].map(HEURISTIC_TO_GROUP)
    df = df.dropna(subset=["winner_group"]).copy()
    return df


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


def load_data(
    path: str = "rollout_labeled_dataset_strong_heuristics_informative_no_ties.csv",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_group_labels(df)
    df = add_derived_features(df)
    return df


# ============================================================
# Model
# ============================================================

def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ============================================================
# Top-k utilities
# ============================================================

def top_k_accuracy(
    y_true: pd.Series,
    proba: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> float:
    top_k_idx = np.argsort(proba, axis=1)[:, -k:]
    top_k_labels = classes[top_k_idx]

    correct = [
        true_label in top_k_labels[i]
        for i, true_label in enumerate(y_true.to_numpy())
    ]

    return float(np.mean(correct))


def prediction_probability_table(
    df_test: pd.DataFrame,
    y_true: pd.Series,
    proba: np.ndarray,
    classes: np.ndarray,
    output_path: str = "group_prediction_probabilities.csv",
) -> pd.DataFrame:
    prob_df = pd.DataFrame(
        proba,
        columns=[f"prob_{c}" for c in classes],
        index=df_test.index,
    )

    top_indices = np.argsort(proba, axis=1)[:, ::-1]

    result = df_test.copy()
    result["true_group"] = y_true.values
    result["top1_group"] = classes[top_indices[:, 0]]
    result["top1_prob"] = proba[np.arange(len(proba)), top_indices[:, 0]]

    if len(classes) >= 2:
        result["top2_group"] = classes[top_indices[:, 1]]
        result["top2_prob"] = proba[np.arange(len(proba)), top_indices[:, 1]]

    if len(classes) >= 3:
        result["top3_group"] = classes[top_indices[:, 2]]
        result["top3_prob"] = proba[np.arange(len(proba)), top_indices[:, 2]]

    result = pd.concat([result, prob_df], axis=1)
    result.to_csv(output_path, index=False)

    print(f"\nSaved prediction probability table: {output_path}")

    return result


def analyze_prediction_confidence(pred_df: pd.DataFrame) -> None:
    print("\n=== Prediction Confidence Analysis ===")

    print("\nTop-1 probability summary:")
    print(pred_df["top1_prob"].describe().round(3))

    if "top2_prob" in pred_df.columns:
        pred_df["top1_top2_gap"] = pred_df["top1_prob"] - pred_df["top2_prob"]

        print("\nTop-1 minus Top-2 probability gap:")
        print(pred_df["top1_top2_gap"].describe().round(3))

        ambiguous = pred_df[pred_df["top1_top2_gap"] < 0.15]
        print("\nAmbiguous predictions (top1-top2 gap < 0.15):")
        print(len(ambiguous), "out of", len(pred_df))
        print(round(100 * len(ambiguous) / len(pred_df), 2), "%")

    print("\nMost common Top-1 predictions:")
    print(pred_df["top1_group"].value_counts())

    if "top2_group" in pred_df.columns:
        print("\nMost common Top-2 predictions:")
        print(pred_df["top2_group"].value_counts())


# ============================================================
# Scenario-level train/test evaluation
# ============================================================

def train_test_by_scenario(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["winner_group"]
    groups = df["scenario"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    df_test = df.iloc[test_idx].copy()

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = model.named_steps["clf"].classes_

    labels = sorted(y.unique())

    print("\n=== Scenario-level Train/Test Split ===")
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Train scenarios:", df.iloc[train_idx]["scenario"].nunique())
    print("Test scenarios:", df.iloc[test_idx]["scenario"].nunique())

    print("\n=== Classification Report: Top-1 Group Prediction ===")
    print(classification_report(y_test, y_pred, labels=labels))

    print("\n=== Confusion Matrix ===")
    print("Labels:", labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    print("\n=== Top-k Accuracy ===")
    top1 = accuracy_score(y_test, y_pred)
    top2 = top_k_accuracy(y_test, proba, classes, k=min(2, len(classes)))
    top3 = top_k_accuracy(y_test, proba, classes, k=min(3, len(classes)))

    print(f"Top-1 accuracy: {top1:.3f}")
    print(f"Top-2 accuracy: {top2:.3f}")
    print(f"Top-3 accuracy: {top3:.3f}")

    pred_df = prediction_probability_table(
        df_test=df_test,
        y_true=y_test,
        proba=proba,
        classes=classes,
        output_path="group_prediction_probabilities.csv",
    )

    analyze_prediction_confidence(pred_df)

    return model, X_train, X_test, y_train, y_test, pred_df


# ============================================================
# Scenario-level cross validation
# ============================================================

def cross_validate_by_scenario(df: pd.DataFrame, n_splits: int = 5) -> None:
    X = df[FEATURE_COLS]
    y = df["winner_group"]
    groups = df["scenario"]

    model = build_model()
    cv = GroupKFold(n_splits=n_splits)

    scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    print("\n=== Scenario-level Cross Validation: Top-1 Accuracy ===")
    print("Scores:", np.round(scores, 3))
    print("Mean accuracy:", round(scores.mean(), 3))
    print("Std:", round(scores.std(), 3))


# ============================================================
# Feature importance
# ============================================================

def plot_feature_importance(
    model: Pipeline,
    save_path: str = "feature_importance_group_model.png",
) -> None:
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_

    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(imp["feature"], imp["importance"])
    plt.title("Feature Importance — Policy Group Classifier")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nSaved feature importance plot: {save_path}")

    print("\n=== Feature Importance ===")
    print(imp.sort_values("importance", ascending=False))


# ============================================================
# 2D visualization
# ============================================================

def plot_decision_space_2d(
    df: pd.DataFrame,
    x_col: str = "mean_tti_to_mean_ttb_ratio",
    y_col: str = "cluster_index",
    save_path: str = "decision_space_group_labels.png",
) -> None:
    labels = sorted(df["winner_group"].unique())

    cmap = plt.get_cmap("tab10")
    label_to_color = {
        label: cmap(i)
        for i, label in enumerate(labels)
    }

    plt.figure(figsize=(9, 7))

    for label in labels:
        subset = df[df["winner_group"] == label]

        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=12,
            alpha=0.55,
            label=label,
            color=label_to_color[label],
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("2D Projection of Policy Group Labels")
    plt.legend(
        title="Policy group",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nSaved 2D decision-space plot: {save_path}")


# ============================================================
# Probability visualization
# ============================================================

def plot_topk_summary(
    pred_df: pd.DataFrame,
    save_path: str = "topk_probability_summary.png",
) -> None:
    if "top2_prob" not in pred_df.columns:
        return

    df = pred_df.copy()
    df["top1_top2_gap"] = df["top1_prob"] - df["top2_prob"]

    plt.figure(figsize=(8, 5))
    plt.hist(df["top1_top2_gap"], bins=30)
    plt.title("Prediction Ambiguity: Top-1 minus Top-2 Probability")
    plt.xlabel("Top-1 probability - Top-2 probability")
    plt.ylabel("Number of states")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nSaved Top-k probability summary plot: {save_path}")


# ============================================================
# Save enriched dataset
# ============================================================

def save_enriched_dataset(
    df: pd.DataFrame,
    output_path: str = "rollout_dataset_grouped_with_features.csv",
) -> None:
    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched dataset: {output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    input_path = "rollout_dataset_compact_heuristics_complex_informative_no_ties.csv"

    df = load_data(input_path)

    print("\n=== Dataset ===")
    print(df.shape)

    print("\n=== Original heuristic label distribution ===")
    print(df["winner"].value_counts())

    print("\n=== Group label distribution ===")
    print(df["winner_group"].value_counts())

    print("\n=== Group label distribution (%) ===")
    print((df["winner_group"].value_counts(normalize=True) * 100).round(2))

    save_enriched_dataset(
        df,
        output_path="rollout_dataset_grouped_with_features.csv",
    )

    model, X_train, X_test, y_train, y_test, pred_df = train_test_by_scenario(df)

    cross_validate_by_scenario(df, n_splits=5)

    plot_feature_importance(
        model,
        save_path="feature_importance_group_model.png",
    )

    plot_decision_space_2d(
        df,
        x_col="mean_tti_to_mean_ttb_ratio",
        y_col="cluster_index",
        save_path="decision_space_group_labels.png",
    )

    plot_topk_summary(
        pred_df,
        save_path="topk_probability_summary.png",
    )


if __name__ == "__main__":
    main()
