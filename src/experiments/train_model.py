from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
    "slack_pressure",
    "urgency_index",
    "density_per_target",
    "feasible_fraction",
]

FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS


# ============================================================
# Data loading and feature engineering
# ============================================================

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stronger derived features for policy-group prediction.

    These features are designed to capture:
    - reachability vs urgency
    - slack pressure
    - feasibility structure
    - spatial density
    """

    df = df.copy()
    eps = 1e-6

    # Replace inf before feature construction where needed
    df = df.replace([np.inf, -np.inf], np.nan)

    # Basic ratio features
    df["mean_tti_to_mean_ttb_ratio"] = df["mean_tti"] / (df["mean_ttb"] + eps)
    df["mean_slack_to_mean_ttb_ratio"] = df["mean_slack"] / (df["mean_ttb"] + eps)
    df["negative_slack_fraction"] = df["count_negative_slack"] / (df["N_active"] + eps)

    # Urgency spread
    df["min_ttb_to_mean_ttb_ratio"] = df["min_ttb"] / (df["mean_ttb"] + eps)
    df["ttb_spread"] = df["mean_ttb"] - df["min_ttb"]

    # Slack pressure:
    # Higher means many targets are already close to infeasible / negative.
    df["slack_pressure"] = -df["mean_slack"]

    # Same idea as negative_slack_fraction, kept with clearer semantic name
    df["urgency_index"] = df["negative_slack_fraction"]

    # Spatial density proxy:
    # Lower cluster_index means denser targets.
    # This feature normalizes by number of active targets.
    df["density_per_target"] = df["cluster_index"] / (df["N_active"] + eps)

    # Feasibility fraction
    df["feasible_fraction"] = 1.0 - df["negative_slack_fraction"]

    # Clean all problematic values after construction
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(1e6)

    return df


def add_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map heuristic-level labels into broader policy groups.
    """

    df = df.copy()
    df["winner_group"] = df["winner"].map(HEURISTIC_TO_GROUP)

    # Drop rows whose winner is not part of the grouped experiment
    df = df.dropna(subset=["winner_group"]).copy()

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
    """
    Random Forest baseline.

    We use class_weight='balanced' because even after grouping,
    the class distribution may remain imbalanced.
    """

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
# Scenario-level train/test evaluation
# ============================================================

def train_test_by_scenario(df: pd.DataFrame):
    """
    Train/test split by scenario.

    This is critical:
    regular random split leaks scenario-specific patterns into both train and test.
    """

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

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    labels = sorted(y.unique())

    print("\n=== Scenario-level Train/Test Split ===")
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Train scenarios:", df.iloc[train_idx]["scenario"].nunique())
    print("Test scenarios:", df.iloc[test_idx]["scenario"].nunique())

    print("\n=== Classification Report: Group Labels ===")
    print(classification_report(y_test, y_pred, labels=labels))

    print("\n=== Confusion Matrix ===")
    print("Labels:", labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    return model, X_train, X_test, y_train, y_test


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

    print("\n=== Scenario-level Cross Validation ===")
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

    plt.figure(figsize=(9, 7))
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
    """
    2D projection of the labeled state space.

    This is not a full decision boundary.
    It is a visual diagnostic to inspect separability.
    """

    labels = sorted(df["winner_group"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}

    x = df[x_col]
    y = df[y_col]
    c = df["winner_group"].map(label_to_id)

    plt.figure(figsize=(9, 7))
    plt.scatter(x, y, c=c, s=10, alpha=0.45)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("2D Projection of Policy Group Labels")

    handles = []
    for label in labels:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                label=label,
                markersize=6,
            )
        )

    plt.legend(
        handles=handles,
        title="Policy group",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nSaved 2D decision-space plot: {save_path}")


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
    input_path = "rollout_labeled_dataset_strong_heuristics_informative_no_ties.csv"

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

    model, X_train, X_test, y_train, y_test = train_test_by_scenario(df)

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


if __name__ == "__main__":
    main()
