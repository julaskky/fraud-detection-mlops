
import json
from pathlib import Path

import pandas as pd


RESULT_DIRECTORY = Path("artifacts/model_results")


MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "mlp",
    "tabnet",
]


def compare_models():
    results = []

    for model_name in MODEL_NAMES:
        result_path = (
            RESULT_DIRECTORY
            / f"{model_name}_validation.json"
        )

        if not result_path.exists():
            print(
                f"Skipping {model_name}: "
                "validation results not found."
            )
            continue

        with open(
            result_path,
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(file)

        results.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "threshold": metrics["threshold"],
        })

    if not results:
        raise RuntimeError(
            "No model results found. Train the models first."
        )

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by=["pr_auc", "f1_score"],
        ascending=False,
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "rank",
        range(1, len(comparison) + 1),
    )

    output_path = (
        RESULT_DIRECTORY / "model_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved:", output_path)

    print(
        "\nBest validation model:",
        comparison.iloc[0]["model"],
    )

    return comparison


if __name__ == "__main__":
    compare_models()
