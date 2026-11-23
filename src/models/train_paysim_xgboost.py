
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(r"D:\datasets\fraudData\PaySimData.csv")

ARTIFACT_DIR = Path("artifacts/paysim")
MODEL_DIR = ARTIFACT_DIR / "models"
RESULT_DIR = ARTIFACT_DIR / "results"

RANDOM_STATE = 42

# Chronological partitions
TRAIN_MAX_STEP = 520
VALIDATION_MAX_STEP = 632

TYPE_MAP = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}

FEATURE_NAMES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",
    "origin_balance_sufficient",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    print("\nLoading PaySim dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataframe = pd.read_csv(
        DATA_PATH,
        usecols=[
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "oldbalanceDest",
            "isFraud",
        ],
        dtype={
            "step": "int16",
            "type": "category",
            "amount": "float32",
            "oldbalanceOrg": "float32",
            "oldbalanceDest": "float32",
            "isFraud": "uint8",
        },
    )

    print(f"Dataset records: {len(dataframe):,}")
    print(f"Fraudulent records: {int(dataframe['isFraud'].sum()):,}")
    print(
        f"Fraud prevalence: "
        f"{dataframe['isFraud'].mean():.6%}"
    )

    return dataframe


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(dataframe):
    amount = dataframe["amount"].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    origin_balance = dataframe["oldbalanceOrg"].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    destination_balance = dataframe["oldbalanceDest"].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    step = dataframe["step"].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    origin_ratio = np.divide(
        amount,
        origin_balance,
        out=np.zeros_like(amount),
        where=origin_balance > 0,
    )

    destination_ratio = np.divide(
        amount,
        destination_balance,
        out=np.zeros_like(amount),
        where=destination_balance > 0,
    )

    origin_sufficient = (
        amount <= origin_balance
    ).astype(np.float32)

    type_codes = dataframe["type"].map(TYPE_MAP)

    if type_codes.isna().any():
        raise ValueError("Unexpected transaction type detected.")

    type_codes = type_codes.to_numpy(dtype=np.int8)

    one_hot = np.zeros(
        (len(dataframe), len(TYPE_MAP)),
        dtype=np.float32,
    )

    one_hot[
        np.arange(len(dataframe)),
        type_codes,
    ] = 1.0

    features = np.empty(
        (len(dataframe), len(FEATURE_NAMES)),
        dtype=np.float32,
    )

    features[:, 0] = step
    features[:, 1] = amount
    features[:, 2] = origin_balance
    features[:, 3] = destination_balance
    features[:, 4] = origin_ratio
    features[:, 5] = destination_ratio
    features[:, 6] = origin_sufficient
    features[:, 7:] = one_hot

    return features


# ============================================================
# CHRONOLOGICAL SPLITTING
# ============================================================

def prepare_splits(dataframe):
    print("\nCreating chronological partitions...")

    train_df = dataframe[
        dataframe["step"] <= TRAIN_MAX_STEP
    ]

    validation_df = dataframe[
        (dataframe["step"] > TRAIN_MAX_STEP)
        & (dataframe["step"] <= VALIDATION_MAX_STEP)
    ]

    test_df = dataframe[
        dataframe["step"] > VALIDATION_MAX_STEP
    ]

    partitions = {
        "Training": train_df,
        "Validation": validation_df,
        "Testing": test_df,
    }

    for name, partition in partitions.items():
        fraud_count = int(partition["isFraud"].sum())

        print(
            f"{name}: {len(partition):,} records | "
            f"Fraud: {fraud_count:,} | "
            f"Prevalence: {partition['isFraud'].mean():.6%}"
        )

        if len(partition) == 0 or fraud_count == 0:
            raise ValueError(
                f"{name} partition is empty or contains no fraud."
            )

    print("\nEngineering training features...")
    X_train = build_features(train_df)
    y_train = train_df["isFraud"].to_numpy(dtype=np.uint8)

    print("Engineering validation features...")
    X_validation = build_features(validation_df)
    y_validation = validation_df["isFraud"].to_numpy(dtype=np.uint8)

    print("Engineering test features...")
    X_test = build_features(test_df)
    y_test = test_df["isFraud"].to_numpy(dtype=np.uint8)

    del train_df, validation_df, test_df, dataframe

    return X_train, y_train, X_validation, y_validation, X_test, y_test


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index])


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

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
            average_precision_score(y_true, probabilities)
        ),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
        ).tolist(),
    }


# ============================================================
# MODEL TRAINING
# ============================================================

def train():
    print("=" * 70)
    print("PAYSIM FRAUD DETECTION - XGBOOST")
    print("=" * 70)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
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

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    print("\nTraining configuration:")
    print(f"Training matrix: {X_train.shape}")
    print(f"Validation matrix: {X_validation.shape}")
    print(f"Testing matrix: {X_test.shape}")
    print(f"Scale positive weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=30,
        tree_method="hist",
        max_bin=256,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")
    print("This may take some time with 6.3 million records.")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation),
        ],
        verbose=25,
    )

    print("\nSelecting decision threshold using validation data...")

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
    print("FINAL TEST RESULTS")
    print("=" * 70)

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    # Save model
    model_path = MODEL_DIR / "xgboost_paysim.json"
    model.save_model(str(model_path))

    # Save feature importance
    importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        RESULT_DIR / "feature_importance.csv",
        index=False,
    )

    # Save experiment results
    results = {
        "dataset": "PaySim",
        "dataset_records": 6362620,
        "target": "isFraud",
        "split_strategy": "chronological",
        "train_max_step": TRAIN_MAX_STEP,
        "validation_max_step": VALIDATION_MAX_STEP,
        "random_state": RANDOM_STATE,
        "best_iteration": int(model.best_iteration),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "features": FEATURE_NAMES,
    }

    result_path = RESULT_DIR / "xgboost_results.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nSaved model:", model_path)
    print("Saved results:", result_path)
    print(
        "Saved feature importance:",
        RESULT_DIR / "feature_importance.csv",
    )

    print("\nXGBoost PaySim training completed.")


if __name__ == "__main__":
    train()
