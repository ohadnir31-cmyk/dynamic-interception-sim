from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from IPython.display import Image, Markdown, display
    IPYTHON_AVAILABLE = True
except Exception:
    IPYTHON_AVAILABLE = False


IMPORTANT_IMAGES = [
    "feature_importance_gini.png",
    "feature_importance_permutation.png",
    "feature_importance_bucket_7-10.png",
    "feature_importance_bucket_11plus.png",
]

BUCKET_IMAGES = [
    "feature_importance_bucket_2-3.png",
    "feature_importance_bucket_4-6.png",
    "feature_importance_bucket_7-10.png",
    "feature_importance_bucket_11plus.png",
]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_markdown(text: str) -> None:
    if IPYTHON_AVAILABLE:
        display(Markdown(text))
    else:
        print(text)


def show_dataframe(df: pd.DataFrame, max_rows: Optional[int] = None) -> None:
    if max_rows is not None:
        df = df.head(max_rows)

    if IPYTHON_AVAILABLE:
        display(df)
    else:
        print(df.to_string(index=False))


def show_image(path: Path) -> None:
    if IPYTHON_AVAILABLE:
        display(Image(filename=str(path)))
    else:
        print(f"Image saved at: {path}")


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"Missing file: {path.name}")
        return None
    return pd.read_csv(path)


def show_available_files(feature_dir: Path) -> None:
    print_section("Available files")

    if not feature_dir.exists():
        print(f"Directory does not exist: {feature_dir}")
        return

    for path in sorted(feature_dir.iterdir()):
        print(path.name)


def show_classification_report(feature_dir: Path) -> None:
    print_section("Classification report")

    path = feature_dir / "classification_report.txt"

    if not path.exists():
        print("classification_report.txt not found")
        return

    print(path.read_text(encoding="utf-8"))


def show_winner_distribution(feature_dir: Path) -> None:
    print_section("Winner distribution used for training")

    path = feature_dir / "winner_distribution_used_for_training.csv"
    df = safe_read_csv(path)

    if df is not None:
        show_dataframe(df)


def show_gini_importance(feature_dir: Path, top_n: int = 20) -> None:
    print_section("Random Forest feature importance")

    csv_path = feature_dir / "feature_importance_gini.csv"
    img_path = feature_dir / "feature_importance_gini.png"

    df = safe_read_csv(csv_path)

    if df is not None:
        show_dataframe(df.head(top_n))

    if img_path.exists():
        show_image(img_path)
    else:
        print("feature_importance_gini.png not found")


def show_permutation_importance(feature_dir: Path, top_n: int = 20) -> None:
    print_section("Permutation feature importance")

    csv_path = feature_dir / "feature_importance_permutation.csv"
    img_path = feature_dir / "feature_importance_permutation.png"

    df = safe_read_csv(csv_path)

    if df is not None:
        show_dataframe(df.head(top_n))

    if img_path.exists():
        show_image(img_path)
    else:
        print("feature_importance_permutation.png not found")


def show_bucket_importance(feature_dir: Path, top_n_per_bucket: int = 10) -> None:
    print_section("Feature importance by active-target bucket")

    csv_path = feature_dir / "feature_importance_by_active_bucket.csv"
    df = safe_read_csv(csv_path)

    if df is not None:
        top_by_bucket = (
            df.sort_values(["N_active_bucket", "importance"], ascending=[True, False])
            .groupby("N_active_bucket")
            .head(top_n_per_bucket)
            .reset_index(drop=True)
        )

        show_dataframe(top_by_bucket)

    for img_name in BUCKET_IMAGES:
        img_path = feature_dir / img_name

        if img_path.exists():
            show_markdown(f"### {img_name}")
            show_image(img_path)
        else:
            print(f"{img_name} not found")


def create_compact_summary(feature_dir: Path, top_n: int = 20) -> Optional[pd.DataFrame]:
    print_section("Compact feature importance summary")

    gini_path = feature_dir / "feature_importance_gini.csv"
    perm_path = feature_dir / "feature_importance_permutation.csv"

    gini = safe_read_csv(gini_path)

    if gini is None:
        return None

    compact = gini[["feature", "importance"]].rename(
        columns={"importance": "gini_importance"}
    )

    perm = safe_read_csv(perm_path)

    if perm is not None:
        compact = compact.merge(
            perm[["feature", "importance_mean", "importance_std"]],
            on="feature",
            how="outer",
        )
    else:
        compact["importance_mean"] = None
        compact["importance_std"] = None

    compact = compact.sort_values("gini_importance", ascending=False)

    output_path = feature_dir / "feature_importance_compact_summary.csv"
    compact.to_csv(output_path, index=False)

    show_dataframe(compact.head(top_n))

    print(f"Saved compact summary to: {output_path}")

    return compact


def display_feature_importance_outputs(
    feature_dir: str | Path,
    top_n: int = 20,
    top_n_per_bucket: int = 10,
) -> None:
    """
    Display feature-importance outputs in Colab or Jupyter.

    This function does not run simulations.
    This function does not train the model.
    It only reads outputs already created by analyze_feature_importance.py.
    """

    feature_dir = Path(feature_dir)

    show_markdown("# Feature Importance Outputs")
    print(f"Feature directory: {feature_dir}")

    show_available_files(feature_dir)
    show_classification_report(feature_dir)
    show_winner_distribution(feature_dir)
    show_gini_importance(feature_dir, top_n=top_n)
    show_permutation_importance(feature_dir, top_n=top_n)
    show_bucket_importance(feature_dir, top_n_per_bucket=top_n_per_bucket)
    create_compact_summary(feature_dir, top_n=top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display feature-importance outputs."
    )

    parser.add_argument(
        "--feature-dir",
        required=True,
        type=str,
        help="Directory containing feature-importance outputs.",
    )

    parser.add_argument(
        "--top-n",
        default=20,
        type=int,
        help="Number of top rows to display.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    display_feature_importance_outputs(
        feature_dir=args.feature_dir,
        top_n=args.top_n,
        top_n_per_bucket=10,
    )


if __name__ == "__main__":
    main()
