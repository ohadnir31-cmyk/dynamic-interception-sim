from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


FEATURE_COLS = [
    "N_active",
    "min_ttb",
    "mean_ttb",
    "min_positive_slack",
    "mean_slack",
    "count_negative_slack",
    "mean_tti",
    "cluster_index",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ניקוי בסיסי
    df = df.copy()
    df = df.replace([float("inf"), float("-inf")], 1e6)

    return df


def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["winner"]

    from sklearn.model_selection import GroupShuffleSplit

    groups = df["scenario"]
    
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    
    X_train = df.iloc[train_idx][FEATURE_COLS]
    y_train = df.iloc[train_idx]["winner"]
    
    X_test = df.iloc[test_idx][FEATURE_COLS]
    y_test = df.iloc[test_idx]["winner"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    return model


def main():
    df = load_data("rollout_labeled_dataset_strong_heuristics_informative_no_ties.csv")

    print("\nDataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["winner"].value_counts())

    model = train_model(df)

    print("\nModel training complete.")


if __name__ == "__main__":
    main()
