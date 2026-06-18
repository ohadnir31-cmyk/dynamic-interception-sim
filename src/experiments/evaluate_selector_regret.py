from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DEFAULT_CANDIDATE_HEURISTICS = ["NI", "FNI", "FMTTB", "MPS", "FCluster"]

DEFAULT_FEATURE_COLUMNS = [
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

SCENARIO_LEVEL_FEATURES = [
    "initial_targets",
    "lambda_arrival",
]


def load_state_dataset(input_dir: Path) -> pd.DataFrame:
    """
    Load state-level rollout data.

    Preferred input:
    large_scale_rollout_states_informative_no_ties.csv

    Fallback:
    large_scale_rollout_states.csv filtered to rows with a unique winner.

    This script does not run simulations. It only reads existing rollout-state files.
    """

    no_ties_path = input_dir / "large_scale_rollout_states_informative_no_ties.csv"
    full_path = input_dir / "large_scale_rollout_states.csv"

    if no_ties_path.exists():
        print(f"Loading no-ties dataset: {no_ties_path}")
        return pd.read_csv(no_ties_path)

    if full_path.exists():
        print(f"Loading full state dataset: {full_path}")
        print("Filtering to rows with a unique winner.")
        df = pd.read_csv(full_path)
        return df[df["winner"] != "TIE"].copy()

    raise FileNotFoundError(
        "Could not find large_scale_rollout_states_informative_no_ties.csv "
        "or large_scale_rollout_states.csv in the input directory."
    )


def infer_candidate_heuristics(df: pd.DataFrame) -> List[str]:
    """
    Infer candidate heuristics from columns such as NI_regret, FNI_regret, etc.
    """

    inferred = []

    for col in df.columns:
        if col.endswith("_regret"):
            h = col.replace("_regret", "")
            if f"{h}_future_intercepted" in df.columns and f"{h}_rank" in df.columns:
                inferred.append(h)

    if inferred:
        return sorted(inferred)

    return DEFAULT_CANDIDATE_HEURISTICS.copy()


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare train/test feature matrices.

    Infinite values are converted to NaN.
    Missing values are filled using train-set medians only.
    """

    existing_features = [c for c in feature_columns if c in train_df.columns]

    if not existing_features:
        raise ValueError("None of the requested feature columns exist in the dataset.")

    missing_features = [c for c in feature_columns if c not in train_df.columns]

    if missing_features:
        print("Warning: missing feature columns ignored:")
        for col in missing_features:
            print(f"  - {col}")

    X_train = train_df[existing_features].copy()
    X_test = test_df[existing_features].copy()

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, X_test, existing_features


def split_by_scenario(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split the dataset by scenario, not by rows.

    This avoids leakage: all states from a scenario are either in train or in test.
    """

    if "scenario" not in df.columns:
        raise ValueError("Dataset must include a 'scenario' column for scenario-level split.")

    scenarios = df["scenario"].unique()

    train_scenarios, test_scenarios = train_test_split(
        scenarios,
        test_size=test_size,
        random_state=random_state,
    )

    train_df = df[df["scenario"].isin(train_scenarios)].copy()
    test_df = df[df["scenario"].isin(test_scenarios)].copy()

    return train_df, test_df, train_scenarios, test_scenarios


def train_selector_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> RandomForestClassifier:
    """
    Train a Random Forest heuristic selector.
    """

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    return model


def add_prediction_and_regret_columns(
    test_df: pd.DataFrame,
    predicted_heuristics: np.ndarray,
    candidate_heuristics: Sequence[str],
) -> pd.DataFrame:
    """
    Add model prediction columns and compute operational regret.

    model_regret = best_future_intercepted - predicted_heuristic_future_intercepted

    Required per-heuristic columns:
    <heuristic>_regret
    <heuristic>_rank
    <heuristic>_future_intercepted
    <heuristic>_future_escaped
    """

    out = test_df.copy()
    out["predicted_heuristic"] = predicted_heuristics
    out["correct_prediction"] = out["predicted_heuristic"] == out["winner"]

    def get_value(row: pd.Series, suffix: str):
        h = row["predicted_heuristic"]
        col = f"{h}_{suffix}"

        if col not in row.index:
            raise KeyError(f"Missing required column: {col}")

        return row[col]

    out["model_regret"] = out.apply(lambda row: get_value(row, "regret"), axis=1)
    out["model_rank"] = out.apply(lambda row: get_value(row, "rank"), axis=1)
    out["model_future_intercepted"] = out.apply(
        lambda row: get_value(row, "future_intercepted"),
        axis=1,
    )

    escaped_cols_exist = all(
        f"{h}_future_escaped" in out.columns for h in candidate_heuristics
    )

    if escaped_cols_exist:
        out["model_future_escaped"] = out.apply(
            lambda row: get_value(row, "future_escaped"),
            axis=1,
        )

    return out


def compute_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_scenarios: Sequence[str],
    test_scenarios: Sequence[str],
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Compute classification and operational evaluation metrics.
    """

    summary: Dict[str, float | int] = {
        "train_scenarios": int(len(train_scenarios)),
        "test_scenarios": int(len(test_scenarios)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "mean_model_regret": float(test_df["model_regret"].mean()),
        "median_model_regret": float(test_df["model_regret"].median()),
        "max_model_regret": float(test_df["model_regret"].max()),
        "zero_regret_rate": float((test_df["model_regret"] == 0).mean()),
        "top1_rate": float((test_df["model_rank"] == 1).mean()),
        "top2_rate": float((test_df["model_rank"] <= 2).mean()),
        "top3_rate": float((test_df["model_rank"] <= 3).mean()),
        "mean_oracle_future_intercepted": float(test_df["best_future_intercepted"].mean()),
        "mean_model_future_intercepted": float(test_df["model_future_intercepted"].mean()),
    }

    if "model_future_escaped" in test_df.columns and "best_future_escaped" in test_df.columns:
        summary["mean_oracle_future_escaped"] = float(test_df["best_future_escaped"].mean())
        summary["mean_model_future_escaped"] = float(test_df["model_future_escaped"].mean())

    return pd.DataFrame([summary])


def summarize_by_bucket(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize model quality by N_active bucket.
    """

    if "N_active_bucket" not in test_df.columns:
        return pd.DataFrame()

    rows = []
    bucket_order = ["1", "2-3", "4-6", "7-10", "11+"]

    for bucket in bucket_order:
        g = test_df[test_df["N_active_bucket"] == bucket].copy()

        if g.empty:
            continue

        rows.append(
            {
                "N_active_bucket": bucket,
                "num_rows": len(g),
                "accuracy": float(g["correct_prediction"].mean()),
                "mean_model_regret": float(g["model_regret"].mean()),
                "median_model_regret": float(g["model_regret"].median()),
                "zero_regret_rate": float((g["model_regret"] == 0).mean()),
                "top1_rate": float((g["model_rank"] == 1).mean()),
                "top2_rate": float((g["model_rank"] <= 2).mean()),
                "top3_rate": float((g["model_rank"] <= 3).mean()),
                "mean_oracle_future_intercepted": float(g["best_future_intercepted"].mean()),
                "mean_model_future_intercepted": float(g["model_future_intercepted"].mean()),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_true_winner(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize model regret by true rollout-based winner class.
    """

    rows = []

    for winner, g in test_df.groupby("winner"):
        rows.append(
            {
                "winner": winner,
                "num_rows": len(g),
                "accuracy": float(g["correct_prediction"].mean()),
                "mean_model_regret": float(g["model_regret"].mean()),
                "median_model_regret": float(g["model_regret"].median()),
                "zero_regret_rate": float((g["model_regret"] == 0).mean()),
                "top2_rate": float((g["model_rank"] <= 2).mean()),
                "top3_rate": float((g["model_rank"] <= 3).mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("num_rows", ascending=False)


def save_classification_outputs(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Save classification report and confusion matrix.
    """

    labels = sorted(y_test.unique())

    report_text = classification_report(y_test, y_pred)

    with open(output_dir / "selector_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "selector_confusion_matrix.csv")


def plot_regret_distribution(test_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save a histogram of model regret.
    """

    plt.figure(figsize=(8, 5))
    plt.hist(test_df["model_regret"], bins=30)
    plt.xlabel("Model regret")
    plt.ylabel("Number of states")
    plt.title("Distribution of selector regret")
    plt.tight_layout()

    path = output_dir / "selector_regret_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {path}")


def plot_bucket_regret(bucket_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Save a bar chart of mean model regret by N_active bucket.
    """

    if bucket_summary.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(bucket_summary["N_active_bucket"], bucket_summary["mean_model_regret"])
    plt.xlabel("Number of active targets")
    plt.ylabel("Mean model regret")
    plt.title("Mean selector regret by active-target bucket")
    plt.tight_layout()

    path = output_dir / "selector_regret_by_active_bucket.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {path}")


def plot_accuracy_by_bucket(bucket_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Save a bar chart of selector accuracy by N_active bucket.
    """

    if bucket_summary.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(bucket_summary["N_active_bucket"], bucket_summary["accuracy"])
    plt.xlabel("Number of active targets")
    plt.ylabel("Accuracy")
    plt.title("Selector accuracy by active-target bucket")
    plt.tight_layout()

    path = output_dir / "selector_accuracy_by_active_bucket.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a learned heuristic selector on held-out scenarios "
            "and compute operational regret."
        )
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
        help="Output directory. Default: input-dir/selector_regret_evaluation",
    )

    parser.add_argument(
        "--test-size",
        default=0.25,
        type=float,
        help="Fraction of scenarios used for testing.",
    )

    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed for scenario-level train/test split and model training.",
    )

    parser.add_argument(
        "--exclude-scenario-features",
        action="store_true",
        help=(
            "Remove scenario-level generator features such as lambda_arrival "
            "and initial_targets from the model."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_dir / "selector_regret_evaluation"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Selector Regret Evaluation ===")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    df = load_state_dataset(input_dir)
    candidate_heuristics = infer_candidate_heuristics(df)

    print(f"Rows loaded: {len(df)}")
    print(f"Candidate heuristics: {candidate_heuristics}")
    print(f"Winner classes: {sorted(df['winner'].unique())}")

    feature_columns = DEFAULT_FEATURE_COLUMNS.copy()

    if args.exclude_scenario_features:
        feature_columns = [
            col for col in feature_columns
            if col not in SCENARIO_LEVEL_FEATURES
        ]
        print("Scenario-level features excluded from model.")

    train_df, test_df, train_scenarios, test_scenarios = split_by_scenario(
        df=df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train, X_test, existing_features = prepare_features(
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
    )

    y_train = train_df["winner"]
    y_test = test_df["winner"]

    print(f"Train scenarios: {len(train_scenarios)}")
    print(f"Test scenarios:  {len(test_scenarios)}")
    print(f"Train rows:      {len(train_df)}")
    print(f"Test rows:       {len(test_df)}")
    print(f"Features used:   {existing_features}")

    model = train_selector_model(
        X_train=X_train,
        y_train=y_train,
        random_state=args.random_state,
    )

    y_pred = model.predict(X_test)

    test_eval = add_prediction_and_regret_columns(
        test_df=test_df,
        predicted_heuristics=y_pred,
        candidate_heuristics=candidate_heuristics,
    )

    summary = compute_summary(
        train_df=train_df,
        test_df=test_eval,
        train_scenarios=train_scenarios,
        test_scenarios=test_scenarios,
        y_test=y_test,
        y_pred=y_pred,
    )

    bucket_summary = summarize_by_bucket(test_eval)
    winner_summary = summarize_by_true_winner(test_eval)

    summary.to_csv(output_dir / "selector_evaluation_summary.csv", index=False)
    test_eval.to_csv(output_dir / "selector_evaluation_test_predictions.csv", index=False)
    bucket_summary.to_csv(output_dir / "selector_evaluation_by_active_bucket.csv", index=False)
    winner_summary.to_csv(output_dir / "selector_evaluation_by_true_winner.csv", index=False)

    save_classification_outputs(y_test, y_pred, output_dir)

    plot_regret_distribution(test_eval, output_dir)
    plot_bucket_regret(bucket_summary, output_dir)
    plot_accuracy_by_bucket(bucket_summary, output_dir)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nBy active-target bucket:")
    print(bucket_summary.to_string(index=False))

    print("\nBy true winner:")
    print(winner_summary.to_string(index=False))

    print("\nDone.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
