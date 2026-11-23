
import json
from pathlib import Path

import joblib

from xgboost import XGBClassifier

from src.models.common import (
    select_threshold,
    calculate_metrics,
)


DATA_PATH = Path(
    "artifacts/complete_case/prepared_data.joblib"
)

MODEL_DIR = Path(
    "artifacts/models_complete_case"
)

RESULT_DIR = Path(
    "artifacts/model_results_complete_case"
)


def train():
    print("=" * 65)
    print("XGBOOST - COMPLETE-CASE DATASET")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Prepared complete-case data not found. "
            "Run python -m src.data.prepare_complete_case first."
        )

    data = joblib.load(DATA_PATH)

    y_train = data["y_train"]

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError(
            "Training data contains no fraudulent transactions."
        )

    scale_pos_weight = negative_count / positive_count

    print("\nTraining records:", len(y_train))
    print("Legitimate transactions:", negative_count)
    print("Fraudulent transactions:", positive_count)
    print("Positive class weight:", scale_pos_weight)

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=40,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")

    model.fit(
        data["X_train"],
        y_train,
        eval_set=[
            (
                data["X_validation"],
                data["y_validation"],
            )
        ],
        verbose=False,
    )

    probabilities = model.predict_proba(
        data["X_validation"]
    )[:, 1]

    threshold = select_threshold(
        data["y_validation"],
        probabilities,
    )

    metrics = calculate_metrics(
        data["y_validation"],
        probabilities,
        threshold,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "xgboost.joblib"

    joblib.dump(
        {
            "model_name": "xgboost_complete_case",
            "model": model,
            "threshold": threshold,
        },
        model_path,
    )

    result_path = (
        RESULT_DIR / "xgboost_validation.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\n========== VALIDATION RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved model:", model_path)
    print("Saved validation results:", result_path)


if __name__ == "__main__":
    train()
