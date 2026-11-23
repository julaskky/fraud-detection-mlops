
import argparse
import json
from pathlib import Path

import joblib

from src.models.common import (
    calculate_metrics,
    load_prepared_data,
)


MODEL_DIRECTORY = Path("artifacts/models")
RESULT_DIRECTORY = Path("artifacts/model_results")


def evaluate(model_name):
    model_path = (
        MODEL_DIRECTORY / f"{model_name}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    bundle = joblib.load(model_path)

    data = load_prepared_data()

    model = bundle["model"]
    threshold = bundle["threshold"]

    print("Evaluating model:", model_name)

    probabilities = model.predict_proba(
        data["X_test"]
    )[:, 1]

    metrics = calculate_metrics(
        data["y_test"],
        probabilities,
        threshold,
    )

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        RESULT_DIRECTORY / "final_test_evaluation.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "selected_model": model_name,
                "metrics": metrics,
            },
            file,
            indent=4,
        )

    print("\n========== FINAL TEST RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved:", result_path)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "logistic_regression",
            "random_forest",
            "xgboost",
            "mlp",
            "tabnet",
        ],
    )

    arguments = parser.parse_args()

    evaluate(arguments.model)
