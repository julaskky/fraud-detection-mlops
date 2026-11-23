
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

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
    print("PAYSIM FRAUD DETECTION - RANDOM FOREST")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the same dataset used by XGBoost
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
    print(f"Training legitimate cases: {int((y_train == 0).sum()):,}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        max_samples=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    print("\nTraining Random Forest...")
    print("This may take some time.")

    model.fit(X_train, y_train)

    print("\nRandom Forest training completed.")

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
    print("RANDOM FOREST FINAL TEST RESULTS")
    print("=" * 70)

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance_path = RESULT_DIR / "random_forest_feature_importance.csv"

    importance.to_csv(
        importance_path,
        index=False,
    )

    print("\nFeature importance:")
    print(importance.to_string(index=False))

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = MODEL_DIR / "random_forest_paysim.joblib"

    joblib.dump(
        {
            "model": model,
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
        "model": "RandomForestClassifier",
        "split_strategy": "chronological",
        "train_max_step": 520,
        "validation_max_step": 632,
        "random_state": RANDOM_STATE,
        "model_parameters": {
            "n_estimators": 200,
            "max_depth": 16,
            "min_samples_split": 10,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "max_samples": 0.5,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "features": FEATURE_NAMES,
    }

    result_path = RESULT_DIR / "random_forest_results.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nSaved model:", model_path)
    print("Saved results:", result_path)
    print("Saved feature importance:", importance_path)

    print("\nRandom Forest PaySim training completed.")


if __name__ == "__main__":
    train()
