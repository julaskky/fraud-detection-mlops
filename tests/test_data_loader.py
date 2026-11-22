from pathlib import Path

import pandas as pd

from src.data.loader import (
    EXPECTED_COLUMNS,
    load_dataset,
    validate_account_age,
    validate_columns,
    validate_dataset,
    validate_target,
    validate_time,
)


DATASET_PATH = Path("data/raw/fraud_detection.csv")


def test_dataset_file_exists():
    assert DATASET_PATH.exists(), (
        f"Dataset not found at {DATASET_PATH}"
    )


def test_dataset_can_be_loaded():
    dataframe = load_dataset(DATASET_PATH)

    assert isinstance(dataframe, pd.DataFrame)
    assert not dataframe.empty


def test_dataset_contains_expected_columns():
    dataframe = load_dataset(DATASET_PATH)

    validate_columns(dataframe)

    assert set(EXPECTED_COLUMNS).issubset(
        dataframe.columns
    )


def test_dataset_contains_at_least_10000_records():
    dataframe = load_dataset(DATASET_PATH)

    assert len(dataframe) >= 10_000


def test_fraudulent_contains_only_binary_values():
    dataframe = load_dataset(DATASET_PATH)

    validate_target(dataframe)

    target_values = set(
        dataframe["Fraudulent"].dropna().unique()
    )

    assert target_values.issubset({0, 1})


def test_transaction_time_is_valid():
    dataframe = load_dataset(DATASET_PATH)

    validate_time(dataframe)


def test_account_age_is_non_negative():
    dataframe = load_dataset(DATASET_PATH)

    validate_account_age(dataframe)


def test_complete_dataset_validation():
    dataframe = load_dataset(DATASET_PATH)

    validate_dataset(dataframe)


def test_duplicate_transaction_ids_are_detected():
    dataframe = load_dataset(DATASET_PATH)

    duplicate_ids = dataframe[
        dataframe["Transaction_ID"].duplicated(
            keep=False
        )
    ]

    assert duplicate_ids["Transaction_ID"].nunique() == 1000


def test_exact_duplicate_rows_are_detected():
    dataframe = load_dataset(DATASET_PATH)

    assert dataframe.duplicated().sum() == 881



def test_duplicate_ids_with_different_details_are_detected():
    dataframe = load_dataset(DATASET_PATH)

    duplicate_rows = dataframe[
        dataframe["Transaction_ID"].duplicated(
            keep=False
        )
    ]

    variation_counts = (
        duplicate_rows
        .groupby("Transaction_ID")
        .nunique(dropna=False)
        .max(axis=1)
    )

    conflicting_ids = variation_counts[
        variation_counts > 1
    ]

    assert len(conflicting_ids) == 119



def test_duplicate_ids_have_consistent_fraud_labels():
    dataframe = load_dataset(DATASET_PATH)

    duplicate_rows = dataframe[
        dataframe["Transaction_ID"].duplicated(
            keep=False
        )
    ]

    label_counts = (
        duplicate_rows
        .groupby("Transaction_ID")["Fraudulent"]
        .nunique()
    )

    assert (label_counts <= 1).all()


def test_fraudulent_column_has_both_classes():
    dataframe = load_dataset(DATASET_PATH)

    target_values = set(
        dataframe["Fraudulent"].dropna().unique()
    )

    assert {0, 1}.issubset(target_values)