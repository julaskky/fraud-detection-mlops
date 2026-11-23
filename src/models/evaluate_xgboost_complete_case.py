
import json
from pathlib import Path

import joblib

from src.models.common import calculate_metrics


DATA_PATH = Path(
    "artifacts/complete_case/prepared_data.joblib"
)

MODEL_PATH = Path(
    "artifacts/models_complete_case/xgboost.joblib"
)

RESULT_PATH = Path(
    "artifacts/model_results_complete_case/final_test_evaluation.json"
)


def main():
    print("=" * 65)
    print("XGBOOST COMPLETE-CASE - FINAL TEST EVALUATION")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Prepared complete-case dataset not found."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Complete-case XGBoost model not found. "
            "Train the model first."
        )

    data = joblib.load(DATA_PATH)
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    threshold = bundle["threshold"]

    print("\nEvaluating XGBoost on untouched test data...")

    probabilities = model.predict_proba(
        data["X_test"]
    )[:, 1]

    metrics = calculate_metrics(
        data["y_test"],
        probabilities,
        threshold,
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\n========== FINAL TEST RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved:", RESULT_PATH)


if __name__ == "__main__":
    main()
