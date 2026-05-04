from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
]

FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_ratio_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(1e6)

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    eps = 1e-6

    df["mean_tti_to_mean_ttb_ratio"] = df["mean_tti"] / (df["mean_ttb"] + eps)
    df["mean_slack_to_mean_ttb_ratio"] = df["mean_slack"] / (df["mean_ttb"] + eps)
    df["negative_slack_fraction"] = df["count_negative_slack"] / (df["N_active"] + eps)

    return df


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_test_by_scenario(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["winner"]
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

    print("\n=== Scenario-level Train/Test Split ===")
    print("Train scenarios:", df.iloc[train_idx]["scenario"].nunique())
    print("Test scenarios:", df.iloc[test_idx]["scenario"].nunique())

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    labels = sorted(y.unique())
    print("Labels:", labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    return model, X_train, X_test, y_train, y_test


def cross_validate_by_scenario(df: pd.DataFrame, n_splits: int = 5) -> None:
    X = df[FEATURE_COLS]
    y = df["winner"]
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


def plot_feature_importance(model: Pipeline, save_path: str = "feature_importance.png") -> None:
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_

    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(imp["feature"], imp["importance"])
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nSaved feature importance plot: {save_path}")
    print("\n=== Feature Importance ===")
    print(imp.sort_values("importance", ascending=False))


def plot_decision_space_2d(
    df: pd.DataFrame,
    x_col: str = "mean_tti_to_mean_ttb_ratio",
    y_col: str = "cluster_index",
    save_path: str = "decision_space_2d.png",
) -> None:
    """
    This is not a true full decision boundary.
    It is a 2D projection of the labeled state space.
    Useful for first visual inspection.
    """

    labels = sorted(df["winner"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}

    x = df[x_col]
    y = df[y_col]
    c = df["winner"].map(label_to_id)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(x, y, c=c, s=10, alpha=0.45)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("2D Projection of Winner Labels")

    handles = []
    for label, idx in label_to_id.items():
        handles.append(
            plt.Line2D(
                [], [],
                marker="o",
                linestyle="",
                label=label,
                markersize=6,
            )
        )

    plt.legend(handles=handles, title="Winner", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"\nSaved 2D decision-space plot: {save_path}")


def save_enriched_dataset(
    df: pd.DataFrame,
    output_path: str = "rollout_dataset_with_ratio_features.csv",
) -> None:
    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched dataset: {output_path}")


def main() -> None:
    input_path = "rollout_labeled_dataset_strong_heuristics_informative_no_ties.csv"

    df = load_data(input_path)

    print("\n=== Dataset ===")
    print(df.shape)

    print("\n=== Class Distribution ===")
    print(df["winner"].value_counts())

    print("\n=== Class Distribution (%) ===")
    print((df["winner"].value_counts(normalize=True) * 100).round(2))

    save_enriched_dataset(
        df,
        output_path="rollout_dataset_strong_heuristics_with_ratio_features.csv",
    )

    model, X_train, X_test, y_train, y_test = train_test_by_scenario(df)

    cross_validate_by_scenario(df, n_splits=5)

    plot_feature_importance(
        model,
        save_path="feature_importance_strong_heuristics.png",
    )

    plot_decision_space_2d(
        df,
        x_col="mean_tti_to_mean_ttb_ratio",
        y_col="cluster_index",
        save_path="decision_space_ratio_vs_cluster.png",
    )


if __name__ == "__main__":
    main()
