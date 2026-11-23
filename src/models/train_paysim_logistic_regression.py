
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.train_paysim_xgboost import (
    load_dataset,
    prepare_splits,
    select_threshold,
    evaluate_model,
    FEATURE_NAMES,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path("artifacts/paysim/models")
RESULT_DIR = Path("artifacts/paysim/results")

RANDOM_STATE = 42


# ============================================================
# TRAINING
# ============================================================

def train():

    print("=" * 70)
    print("PAYSIM FRAUD DETECTION - LOGISTIC REGRESSION")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_splits(dataframe)

    print("\nTraining configuration:")
    print(f"Training records: {len(y_train):,}")
    print(f"Validation records: {len(y_validation):,}")
    print(f"Test records: {len(y_test):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")

    # ========================================================
    # FEATURE SCALING
    # ========================================================

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(
        np.float32,
        copy=False,
    )

    X_validation = scaler.transform(X_validation).astype(
        np.float32,
        copy=False,
    )

    X_test = scaler.transform(X_test).astype(
        np.float32,
        copy=False,
    )

    print("Feature scaling completed.")

    # ========================================================
    # MODEL
    # ========================================================

    model = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=100,
        tol=1e-3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    print("\nTraining Logistic Regression...")
    print("This may take some time on 6.3 million records.")

    model.fit(X_train, y_train)

    print("\nTraining completed.")

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nEvaluating validation data...")

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    threshold = select_threshold(
        y_validation,
        validation_probabilities,
    )

    validation_metrics = evaluate_model(
        y_validation,
        validation_probabilities,
        threshold,
    )

    del validation_probabilities

    # ========================================================
    # TEST
    # ========================================================

    print("\nEvaluating untouched test data...")

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = evaluate_model(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION FINAL TEST RESULTS")
    print("=" * 70)

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    # ========================================================
    # COEFFICIENT IMPORTANCE
    # ========================================================

    coefficients = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "coefficient": model.coef_[0],
        "absolute_coefficient": np.abs(model.coef_[0]),
    }).sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    coefficients.to_csv(
        RESULT_DIR / "logistic_regression_coefficients.csv",
        index=False,
    )

    print("\nFeature coefficients:")
    print(coefficients.to_string(index=False))

    # ========================================================
    # SAVE MODEL AND SCALER
    # ========================================================

    model_path = MODEL_DIR / "logistic_regression_paysim.joblib"

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "threshold": threshold,
            "feature_names": FEATURE_NAMES,
        },
        model_path,
        compress=3,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "dataset": "PaySim",
        "model": "LogisticRegression",
        "split_strategy": "chronological",
        "train_max_step": 520,
        "validation_max_step": 632,
        "random_state": RANDOM_STATE,
        "model_parameters": {
            "solver": "saga",
            "penalty": "l2",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 100,
            "tol": 1e-3,
        },
        "iterations": model.n_iter_.tolist(),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "features": FEATURE_NAMES,
    }

    result_path = RESULT_DIR / "logistic_regression_results.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nSaved model:", model_path)
    print(
        "Saved results:",
        result_path,
    )
    print(
        "Saved coefficients:",
        RESULT_DIR / "logistic_regression_coefficients.csv",
    )

    print("\nLogistic Regression PaySim training completed.")


if __name__ == "__main__":
    train()
