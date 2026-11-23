
import json
from pathlib import Path

import joblib
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


DATA_PATH = Path(
    "artifacts/complete_case/prepared_data.joblib"
)

RESULT_DIR = Path(
    "artifacts/model_results_complete_case"
)

RESULT_PATH = (
    RESULT_DIR / "label_permutation_test.json"
)

RANDOM_STATE = 42


def create_model():
    """Create XGBoost using the existing project configuration."""

    return XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        scale_pos_weight=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=40,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def calculate_auc_metrics(y_true, probabilities):
    return {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
    }


def train_and_evaluate(
    name,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    print(f"\nTraining {name}...")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            f"{name}: training labels must contain both classes."
        )

    model = create_model()

    model.set_params(
        scale_pos_weight=negative_count / positive_count
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_auc_metrics(
        y_test,
        probabilities,
    )

    metrics["best_iteration"] = int(
        model.best_iteration
    )

    print(f"\n{name} test results:")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    return model, metrics


def main():
    print("=" * 70)
    print("XGBOOST TARGET-LABEL PERMUTATION TEST")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Complete-case prepared data not found. "
            "Run complete-case preparation first."
        )

    print("\nLoading complete-case prepared data...")

    data = joblib.load(DATA_PATH)

    X_train = data["X_train"]
    y_train = np.asarray(data["y_train"]).astype(int)

    X_validation = data["X_validation"]
    y_validation = np.asarray(
        data["y_validation"]
    ).astype(int)

    X_test = data["X_test"]
    y_test = np.asarray(data["y_test"]).astype(int)

    print("Training records:", len(y_train))
    print("Validation records:", len(y_validation))
    print("Test records:", len(y_test))
    print("Test fraud prevalence:", round(y_test.mean(), 5))

    # --------------------------------------------------
    # EXPERIMENT 1: REAL LABELS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: REAL LABELS")
    print("=" * 70)

    real_model, real_metrics = train_and_evaluate(
        name="Real-label XGBoost",
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
    )

    # --------------------------------------------------
    # EXPERIMENT 2: SHUFFLED LABELS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SHUFFLED LABELS")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_STATE)

    shuffled_y_train = rng.permutation(y_train)
    shuffled_y_validation = rng.permutation(
        y_validation
    )

    print("Training labels randomly shuffled.")
    print("Validation labels randomly shuffled.")
    print("Test labels remain unchanged.")

    shuffled_model, shuffled_metrics = train_and_evaluate(
        name="Shuffled-label XGBoost",
        X_train=X_train,
        y_train=shuffled_y_train,
        X_validation=X_validation,
        y_validation=shuffled_y_validation,
        X_test=X_test,
        y_test=y_test,
    )

    # --------------------------------------------------
    # COMPARISON
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL PERMUTATION TEST COMPARISON")
    print("=" * 70)

    comparison = {
        "real_labels": real_metrics,
        "shuffled_labels": shuffled_metrics,
        "differences": {
            "roc_auc": (
                real_metrics["roc_auc"]
                - shuffled_metrics["roc_auc"]
            ),
            "pr_auc": (
                real_metrics["pr_auc"]
                - shuffled_metrics["pr_auc"]
            ),
        },
        "test_fraud_prevalence": float(y_test.mean()),
        "random_state": RANDOM_STATE,
    }

    print(
        f"\n{'Metric':<15}"
        f"{'Real Labels':<18}"
        f"{'Shuffled Labels':<18}"
        f"{'Difference':<15}"
    )

    for metric in ["roc_auc", "pr_auc"]:

        real_value = real_metrics[metric]
        shuffled_value = shuffled_metrics[metric]

        difference = real_value - shuffled_value

        print(
            f"{metric:<15}"
            f"{real_value:<18.6f}"
            f"{shuffled_value:<18.6f}"
            f"{difference:<15.6f}"
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            indent=4,
        )

    print("\nSaved comparison:", RESULT_PATH)

    print("\nPermutation test completed.")


if __name__ == "__main__":
    main()
