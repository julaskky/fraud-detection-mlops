
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_DIRECTORY = Path("artifacts/models")
RESULT_DIRECTORY = Path("artifacts/model_results")


def load_prepared_data():
    """Load the prepared datasets generated in Phase 2."""

    path = Path("artifacts/prepared_data.joblib")

    if not path.exists():
        raise FileNotFoundError(
            "Prepared data not found. Run Phase 2 first."
        )

    return joblib.load(path)


def to_dense_float32(data):
    """Convert sparse or dense input to a float32 NumPy array."""

    if hasattr(data, "toarray"):
        data = data.toarray()

    return np.asarray(data, dtype=np.float32)


def select_threshold(y_true, probabilities):
    """
    Select a classification threshold using validation F1.

    The test set is not used for threshold selection.
    """

    thresholds = np.linspace(0.10, 0.90, 81)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
):
    """Calculate binary fraud classification metrics."""

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).reshape(-1)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "threshold": float(threshold),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                predictions,
            ).tolist()
        ),
    }


def save_validation_result(
    model_name,
    model,
    X_validation,
    y_validation,
):
    """
    Select the model's threshold on validation data
    and save the model and validation metrics.
    """

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    threshold = select_threshold(
        y_validation,
        probabilities,
    )

    metrics = calculate_metrics(
        y_validation,
        probabilities,
        threshold,
    )

    model_path = (
        MODEL_DIRECTORY / f"{model_name}.joblib"
    )

    joblib.dump(
        {
            "model_name": model_name,
            "model": model,
            "threshold": threshold,
        },
        model_path,
    )

    result_path = (
        RESULT_DIRECTORY
        / f"{model_name}_validation.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(f"\nModel: {model_name}")
    print("Validation metrics:")

    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"{key}: {value}")

    print("Saved model:", model_path)
    print("Saved validation results:", result_path)

    return metrics
