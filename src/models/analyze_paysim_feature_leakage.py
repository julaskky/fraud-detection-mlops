
"""
PAYSIM FEATURE LEAKAGE AND ROBUSTNESS ANALYSIS

Experiments:
1. Baseline: all features.
2. Remove transaction type features.
3. Restrict the dataset to CASH_OUT and TRANSFER.
4. Remove origin_balance_sufficient.
5. Remove both balance-derived ratio features.
6. Remove all balance-derived features.
7. Evaluate performance separately for CASH_OUT and TRANSFER.

Models:
- XGBoost
- Random Forest

Important:
- Uses existing chronological partitions.
- Uses validation data for threshold selection.
- Evaluates final performance on the test set.
- Does not overwrite existing trained model artifacts.
"""

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from src.models.train_paysim_xgboost import (
    load_dataset,
    prepare_splits,
    FEATURE_NAMES,
    TRAIN_MAX_STEP,
    VALIDATION_MAX_STEP,
)


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

RESULTS_DIR = Path("artifacts/paysim/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = RESULTS_DIR / "feature_leakage_analysis.json"
OUTPUT_CSV = RESULTS_DIR / "feature_leakage_analysis.csv"

FEATURE_NAMES = list(FEATURE_NAMES)

TYPE_FEATURES = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

RATIO_FEATURES = [
    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",
]

ORIGIN_BALANCE_FEATURE = "origin_balance_sufficient"

TRANSACTION_TYPES = ["CASH_OUT", "TRANSFER"]


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

def create_xgboost_model(y_train):
    """Create XGBoost using the existing baseline configuration."""

    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))

    return XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=negatives / max(positives, 1),
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=30,
        tree_method="hist",
        max_bin=256,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def create_random_forest_model():
    """Create Random Forest using the existing baseline configuration."""

    return RandomForestClassifier(
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
        verbose=0,
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataset_splits(dataframe):
    """
    Use the existing PaySim feature engineering and chronological
    partitioning implementation.

    IMPORTANT:
    prepare_splits() returns NumPy arrays, not pandas DataFrames.
    """

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_splits(dataframe)

    # Explicitly convert to NumPy arrays.
    X_train = np.asarray(X_train, dtype=np.float32)
    X_validation = np.asarray(X_validation, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    y_train = np.asarray(y_train, dtype=np.uint8)
    y_validation = np.asarray(y_validation, dtype=np.uint8)
    y_test = np.asarray(y_test, dtype=np.uint8)

    # Validate feature dimensions.
    expected_features = len(FEATURE_NAMES)

    for name, X in [
        ("Training", X_train),
        ("Validation", X_validation),
        ("Test", X_test),
    ]:
        if X.ndim != 2:
            raise ValueError(
                f"{name} feature matrix must be 2-dimensional."
            )

        if X.shape[1] != expected_features:
            raise ValueError(
                f"{name} feature matrix has {X.shape[1]} columns, "
                f"but FEATURE_NAMES contains {expected_features} names."
            )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


def get_raw_test_partition(dataframe):
    """Extract the original chronological test partition."""

    return dataframe[
        dataframe["step"] > VALIDATION_MAX_STEP
    ].copy()


def validate_test_alignment(test_dataframe, y_test):
    """Ensure raw test transaction types align with engineered labels."""

    if len(test_dataframe) != len(y_test):
        raise ValueError(
            "Test alignment error: raw test records and engineered "
            "test labels have different lengths."
        )

    return (
        test_dataframe["type"]
        .astype(str)
        .reset_index(drop=True)
        .to_numpy()
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_indices(selected_features):
    """Convert feature names into NumPy column indices."""

    missing = [
        feature
        for feature in selected_features
        if feature not in FEATURE_NAMES
    ]

    if missing:
        raise ValueError(
            f"Unknown features requested: {missing}"
        )

    indices = [
        FEATURE_NAMES.index(feature)
        for feature in selected_features
    ]

    if not indices:
        raise ValueError("No features selected for experiment.")

    return indices


def select_features(X, selected_features):
    """Select columns from a NumPy feature matrix."""

    indices = get_feature_indices(selected_features)

    return X[:, indices]


def define_experiments():
    """Return the feature-ablation experiment configurations."""

    baseline = list(FEATURE_NAMES)

    return {
        "baseline_all_features": baseline,

        "without_transaction_type": [
            feature
            for feature in baseline
            if feature not in TYPE_FEATURES
        ],

        "without_origin_balance_sufficient": [
            feature
            for feature in baseline
            if feature != ORIGIN_BALANCE_FEATURE
        ],

        "without_balance_ratios": [
            feature
            for feature in baseline
            if feature not in RATIO_FEATURES
        ],

        "without_all_balance_derived_features": [
            feature
            for feature in baseline
            if feature not in (
                RATIO_FEATURES + [ORIGIN_BALANCE_FEATURE]
            )
        ],
    }


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def select_validation_threshold(y_true, probabilities):
    """
    Select the threshold that maximizes validation F1.

    The test set is never used for threshold selection.
    """

    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.uint8)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification metrics and confusion matrix."""

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

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

        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else None
        ),

        "pr_auc": (
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            )
            if np.sum(y_true == 1) > 0
            else None
        ),

        "threshold": float(threshold),

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),

        "fraud_cases": int(np.sum(y_true == 1)),
        "records": int(len(y_true)),
        "fraud_prevalence": float(np.mean(y_true)),
    }


# ============================================================
# MODEL TRAINING
# ============================================================

def fit_model(
    model_name,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Train one model using the specified training population."""

    if model_name == "xgboost":

        model = create_xgboost_model(y_train)

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_validation, y_validation)
            ],
            verbose=False,
        )

    elif model_name == "random_forest":

        model = create_random_forest_model()

        model.fit(
            X_train,
            y_train,
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return model


# ============================================================
# STANDARD FEATURE-ABLATION EXPERIMENTS
# ============================================================

def run_experiment(
    model_name,
    experiment_name,
    selected_features,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    """Train and evaluate a model for a feature-ablation experiment."""

    print("\n" + "=" * 75)
    print(f"MODEL: {model_name.upper()}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"FEATURE COUNT: {len(selected_features)}")
    print("=" * 75)

    # Select features using NumPy column indices.
    X_train_exp = select_features(
        X_train,
        selected_features,
    )

    X_validation_exp = select_features(
        X_validation,
        selected_features,
    )

    X_test_exp = select_features(
        X_test,
        selected_features,
    )

    print("Training model...")

    model = fit_model(
        model_name,
        X_train_exp,
        y_train,
        X_validation_exp,
        y_validation,
    )

    print("Selecting threshold using validation data...")

    validation_probabilities = model.predict_proba(
        X_validation_exp
    )[:, 1]

    threshold = select_validation_threshold(
        y_validation,
        validation_probabilities,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        threshold,
    )

    print("Evaluating untouched test data...")

    test_probabilities = model.predict_proba(
        X_test_exp
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\nTest results:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    result = {
        "model": model_name,
        "experiment": experiment_name,
        "features": selected_features,
        "feature_count": len(selected_features),
        "validation": validation_metrics,
        "test": test_metrics,
    }

    del model
    del X_train_exp
    del X_validation_exp
    del X_test_exp

    gc.collect()

    return result


# ============================================================
# RESTRICTED TRANSACTION-TYPE EXPERIMENT
# ============================================================

def run_restricted_transaction_experiment(
    dataframe,
    model_name,
):
    """
    Train using only CASH_OUT and TRANSFER transactions.

    Chronological partitioning is performed again after filtering.
    """

    print("\n" + "=" * 75)
    print(
        f"{model_name.upper()} - CASH_OUT AND TRANSFER ONLY"
    )
    print("=" * 75)

    restricted = dataframe[
        dataframe["type"].astype(str).isin(
            TRANSACTION_TYPES
        )
    ].copy()

    if restricted.empty:
        raise ValueError(
            "No CASH_OUT or TRANSFER transactions found."
        )

    print(f"Restricted records: {len(restricted):,}")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_dataset_splits(restricted)

    print(
        f"Training: {len(y_train):,} | "
        f"Fraud: {int(np.sum(y_train)):,}"
    )

    print(
        f"Validation: {len(y_validation):,} | "
        f"Fraud: {int(np.sum(y_validation)):,}"
    )

    print(
        f"Test: {len(y_test):,} | "
        f"Fraud: {int(np.sum(y_test)):,}"
    )

    result = run_experiment(
        model_name=model_name,
        experiment_name="cash_out_transfer_only",
        selected_features=FEATURE_NAMES,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
    )

    del restricted
    del X_train, X_validation, X_test
    del y_train, y_validation, y_test

    gc.collect()

    return result


# ============================================================
# TRANSACTION-TYPE-SPECIFIC EVALUATION
# ============================================================

def evaluate_by_transaction_type(
    dataframe,
    model_name,
):
    """
    Train on all transaction types, then evaluate CASH_OUT and
    TRANSFER separately on the chronological test partition.

    The same validation-selected threshold is used for both groups.
    """

    print("\n" + "=" * 75)
    print(
        f"{model_name.upper()} - TYPE-SPECIFIC TEST EVALUATION"
    )
    print("=" * 75)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_dataset_splits(dataframe)

    test_dataframe = get_raw_test_partition(dataframe)

    test_types = validate_test_alignment(
        test_dataframe,
        y_test,
    )

    print("Training model on the complete training population...")

    model = fit_model(
        model_name,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    threshold = select_validation_threshold(
        y_validation,
        validation_probabilities,
    )

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    type_results = {}

    for transaction_type in TRANSACTION_TYPES:

        print(f"\nEvaluating {transaction_type}...")

        mask = test_types == transaction_type

        if not np.any(mask):
            print("No transactions found. Skipping.")
            continue

        type_y = y_test[mask]
        type_probabilities = test_probabilities[mask]

        metrics = calculate_metrics(
            type_y,
            type_probabilities,
            threshold,
        )

        type_results[transaction_type] = metrics

        print(json.dumps(metrics, indent=4))

    result = {
        "model": model_name,
        "experiment": "type_specific_evaluation",
        "threshold": float(threshold),
        "transaction_types": type_results,
    }

    del model
    del X_train, X_validation, X_test
    del y_train, y_validation, y_test

    gc.collect()

    return result


# ============================================================
# RESULTS EXPORT
# ============================================================

def save_results(all_results):
    """Save complete JSON results and a flattened CSV summary."""

    output = {
        "dataset": "PaySim",
        "analysis": "Feature leakage and robustness analysis",
        "split_strategy": {
            "training_max_step": int(TRAIN_MAX_STEP),
            "validation_max_step": int(VALIDATION_MAX_STEP),
            "test_steps": f">{VALIDATION_MAX_STEP}",
        },
        "experiments": all_results,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    rows = []

    for result in all_results:

        # Standard and restricted experiments.
        if "test" in result:

            metrics = result["test"]

            rows.append({
                "model": result["model"],
                "experiment": result["experiment"],
                "feature_count": result["feature_count"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "threshold": metrics["threshold"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "fraud_cases": metrics["fraud_cases"],
                "records": metrics["records"],
            })

        # Type-specific experiments.
        elif "transaction_types" in result:

            for transaction_type, metrics in (
                result["transaction_types"].items()
            ):

                rows.append({
                    "model": result["model"],
                    "experiment": (
                        f"type_specific_{transaction_type}"
                    ),
                    "feature_count": len(FEATURE_NAMES),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "threshold": metrics["threshold"],
                    "true_positives": metrics["true_positives"],
                    "false_positives": metrics["false_positives"],
                    "false_negatives": metrics["false_negatives"],
                    "fraud_cases": metrics["fraud_cases"],
                    "records": metrics["records"],
                })

    results_dataframe = pd.DataFrame(rows)

    results_dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    return results_dataframe


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 75)
    print("PAYSIM FEATURE LEAKAGE AND ROBUSTNESS ANALYSIS")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\nLoading PaySim dataset...")

    dataframe = load_dataset()

    print(f"Dataset records: {len(dataframe):,}")

    print("\nPreparing chronological partitions...")

    print(
        f"Training maximum step: {TRAIN_MAX_STEP}"
    )

    print(
        f"Validation maximum step: {VALIDATION_MAX_STEP}"
    )

    # --------------------------------------------------------
    # 2. ENGINEER FEATURES
    # --------------------------------------------------------

    print("\nEngineering features...")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_dataset_splits(dataframe)

    print(f"Features available: {len(FEATURE_NAMES)}")

    print(
        f"Training: {len(y_train):,} records | "
        f"Fraud: {int(np.sum(y_train)):,}"
    )

    print(
        f"Validation: {len(y_validation):,} records | "
        f"Fraud: {int(np.sum(y_validation)):,}"
    )

    print(
        f"Testing: {len(y_test):,} records | "
        f"Fraud: {int(np.sum(y_test)):,}"
    )

    # --------------------------------------------------------
    # 3. FEATURE ABLATION EXPERIMENTS
    # --------------------------------------------------------

    experiments = define_experiments()

    all_results = []

    for model_name in ["xgboost", "random_forest"]:

        for experiment_name, selected_features in (
            experiments.items()
        ):

            result = run_experiment(
                model_name=model_name,
                experiment_name=experiment_name,
                selected_features=selected_features,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                X_test=X_test,
                y_test=y_test,
            )

            all_results.append(result)

    # Release the full matrices before the remaining experiments.
    del X_train, X_validation, X_test
    del y_train, y_validation, y_test

    gc.collect()

    # --------------------------------------------------------
    # 4. RESTRICTED TRANSACTION EXPERIMENT
    # --------------------------------------------------------

    for model_name in ["xgboost", "random_forest"]:

        result = run_restricted_transaction_experiment(
            dataframe=dataframe,
            model_name=model_name,
        )

        all_results.append(result)

    # --------------------------------------------------------
    # 5. TRANSACTION-TYPE-SPECIFIC EVALUATION
    # --------------------------------------------------------

    for model_name in ["xgboost", "random_forest"]:

        result = evaluate_by_transaction_type(
            dataframe=dataframe,
            model_name=model_name,
        )

        all_results.append(result)

    # --------------------------------------------------------
    # 6. SAVE RESULTS
    # --------------------------------------------------------

    print("\nSaving experiment results...")

    results_dataframe = save_results(all_results)

    # --------------------------------------------------------
    # 7. DISPLAY SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FEATURE LEAKAGE ANALYSIS COMPLETED")
    print("=" * 75)

    summary_columns = [
        "model",
        "experiment",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]

    print(
        results_dataframe[summary_columns].to_string(
            index=False
        )
    )

    print(f"\nJSON results: {OUTPUT_JSON}")
    print(f"CSV results: {OUTPUT_CSV}")

    print("\nExisting model artifacts were not modified.")


if __name__ == "__main__":
    main()
