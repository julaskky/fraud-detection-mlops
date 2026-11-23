
import numpy as np

from src.models.common import (
    calculate_metrics,
    select_threshold,
    to_dense_float32,
)


def test_threshold_selection_returns_valid_threshold():
    y_true = np.array([0, 0, 1, 1, 0, 1])

    probabilities = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
        0.30,
        0.70,
    ])

    threshold = select_threshold(
        y_true,
        probabilities,
    )

    assert 0.10 <= threshold <= 0.90


def test_metrics_returns_expected_fields():
    y_true = np.array([0, 0, 1, 1])

    probabilities = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
    ])

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.50,
    )

    expected_fields = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "threshold",
        "confusion_matrix",
    ]

    for field in expected_fields:
        assert field in metrics


def test_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])

    probabilities = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
    ])

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.50,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_dense_conversion_returns_float32():
    data = np.array([
        [1, 2],
        [3, 4],
    ])

    converted = to_dense_float32(data)

    assert converted.dtype == np.float32
    assert converted.shape == (2, 2)
