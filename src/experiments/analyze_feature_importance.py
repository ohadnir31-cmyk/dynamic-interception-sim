from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
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
    "initial_targets",
    "lambda_arrival",
]


def load_dataset(input_dir: Path) -> pd.DataFrame:
    no_ties_file = input_dir / "large_scale_rollout_states_informative_no_ties.csv"
    full_file = input_dir / "large_scale_rollout_states.csv"

    if no_ties_file.exists():
        print(f"Loading: {no_ties_file}")
        return pd.read_csv(no_ties_file)

    if full_file.exists():
        print(f"Loading: {full_file}")
        print("Filtering to rows with a unique winner.")
        df = pd.read_csv(full_file)
        return df[df["winner"] != "TIE"].copy()

    raise FileNotFoundError(
        f"No state-label file found in {input_dir}. "
        "Expected large_scale_rollout_states_informative_no_ties.csv "
        "or large_scale_rollout_states.csv."
    )


def prepare_data(df: pd.DataFrame, feature_cols: List[str]):
    existing_cols = [c for c in feature_cols if c in df.columns]

    if not existing_cols:
        raise ValueError("None of the requested feature columns exist in the dataset.")

    X = df[existing_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    y = df["winner"].copy()

    return X, y, existing_cols


def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred


def save_model_quality(y_test, y_pred, output_dir: Path):
    labels = sorted(y_test.unique())
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    print("\nModel quality")
    print("-------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)


def plot_barh(df: pd.DataFrame, value_col: str, title: str, xlabel: str, output_path: Path):
    plot_df = df.sort_values(value_col)

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["feature"], plot_df[value_col])
    plt.xlabel(xlabel)
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {output_path}")


def save_gini_importance(model, feature_cols: List[str], output_dir: Path, top_n: int):
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(output_dir / "feature_importance_gini.csv", index=False)

    top = importance.head(top_n)

    plot_barh(
        df=top,
        value_col="importance",
        title="Global feature importance for heuristic selection",
        xlabel="Feature importance",
        output_path=output_dir / "feature_importance_gini.png",
    )

    return importance


def save_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    top_n: int,
    n_repeats: int,
):
    print("\nComputing permutation importance...")
    print("This can take a few minutes.")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    importance.to_csv(output_dir / "feature_importance_permutation.csv", index=False)

    top = importance.head(top_n).rename(columns={"importance_mean": "importance"})

    plot_barh(
        df=top,
        value_col="importance",
        title="Permutation feature importance for heuristic selection",
        xlabel="Mean decrease in accuracy",
        output_path=output_dir / "feature_importance_permutation.png",
    )

    return importance


def save_bucket_importance(df: pd.DataFrame, feature_cols: List[str], output_dir: Path, top_n: int):
    if "N_active_bucket" not in df.columns:
        print("Skipping bucket analysis: N_active_bucket column does not exist.")
        return

    rows = []
    bucket_order = ["2-3", "4-6", "7-10", "11+"]

    for bucket in bucket_order:
        df_bucket = df[df["N_active_bucket"] == bucket].copy()

        if len(df_bucket) < 100:
            print(f"Skipping bucket {bucket}: only {len(df_bucket)} rows.")
            continue

        if df_bucket["winner"].nunique() < 2:
            print(f"Skipping bucket {bucket}: only one winner class.")
            continue

        print(f"\nTraining bucket model: N_active_bucket={bucket}, rows={len(df_bucket)}")

        X, y, existing_cols = prepare_data(df_bucket, feature_cols)
        model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)

        acc = accuracy_score(y_test, y_pred)

        importance = pd.DataFrame(
            {
                "feature": existing_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        for _, r in importance.iterrows():
            rows.append(
                {
                    "N_active_bucket": bucket,
                    "accuracy": acc,
                    "num_rows": len(df_bucket),
                    "feature": r["feature"],
                    "importance": r["importance"],
                }
            )

        top = importance.head(top_n)
        safe_bucket = bucket.replace("+", "plus")

        plot_barh(
            df=top,
            value_col="importance",
            title=f"Feature importance for N_active={bucket}",
            xlabel="Feature importance",
            output_path=output_dir / f"feature_importance_bucket_{safe_bucket}.png",
        )

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(output_dir / "feature_importance_by_active_bucket.csv", index=False)


def save_winner_distribution(df: pd.DataFrame, output_dir: Path):
    dist = (
        df["winner"]
        .value_counts()
        .rename_axis("winner")
        .reset_index(name="count")
    )
    dist["share"] = dist["count"] / dist["count"].sum()

    dist.to_csv(output_dir / "winner_distribution_used_for_training.csv", index=False)

    print("\nWinner distribution")
    print("-------------------")
    print(dist)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze feature importance from existing rollout-state results."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing large_scale_rollout_states files.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Output directory. Default: input-dir/feature_importance",
    )

    parser.add_argument(
        "--top-n",
        default=15,
        type=int,
        help="Number of top features to show in plots.",
    )

    parser.add_argument(
        "--permutation-repeats",
        default=10,
        type=int,
        help="Number of permutation-importance repeats.",
    )

    parser.add_argument(
        "--skip-permutation",
        action="store_true",
        help="Skip permutation importance for a faster run.",
    )

    parser.add_argument(
        "--skip-buckets",
        action="store_true",
        help="Skip feature importance by N_active bucket.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Feature Importance Analysis ===")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    df = load_dataset(input_dir)

    print(f"\nRows loaded: {len(df)}")
    print(f"Winner classes: {sorted(df['winner'].unique())}")

    save_winner_distribution(df, output_dir)

    X, y, existing_cols = prepare_data(df, FEATURE_COLUMNS)

    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)

    save_model_quality(y_test, y_pred, output_dir)

    gini_importance = save_gini_importance(
        model=model,
        feature_cols=existing_cols,
        output_dir=output_dir,
        top_n=args.top_n,
    )

    if not args.skip_permutation:
        save_permutation_importance(
            model=model,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
            top_n=args.top_n,
            n_repeats=args.permutation_repeats,
        )

    if not args.skip_buckets:
        save_bucket_importance(
            df=df,
            feature_cols=FEATURE_COLUMNS,
            output_dir=output_dir,
            top_n=args.top_n,
        )

    print("\nTop global features")
    print("-------------------")
    print(gini_importance.head(args.top_n))

    print("\nDone.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
