
import pandas as pd
import pytest

from src.data.inspection import inspect_dataset
from src.data.cleaning import clean_dataset
from src.features.engineering import engineer_features
from src.preprocessing.splitting import split_dataset
from src.preprocessing.pipeline import build_preprocessor


@pytest.fixture
def sample_dataset():
    records = []

    for i in range(40):
        records.append({
            "Transaction_ID": f"T{i}",
            "User_ID": f"U{i % 10}",
            "Transaction_Amount": float(100 + i),
            "Transaction_Type": (
                "Online" if i % 2 == 0 else "POS"
            ),
            "Time_of_Transaction": i % 24,
            "Device_Used": (
                "Mobile" if i % 2 == 0 else "Desktop"
            ),
            "Location": (
                "Urban" if i % 2 == 0 else "Rural"
            ),
            "Previous_Fraudulent_Transactions": i % 3,
            "Account_Age": 30 + i,
            "Number_of_Transactions_Last_24H": i % 8,
            "Payment_Method": (
                "Card" if i % 2 == 0 else "Bank Transfer"
            ),
            "Fraudulent": i % 2,
        })

    dataframe = pd.DataFrame(records)

    dataframe.loc[0, "Transaction_Amount"] = None
    dataframe.loc[1, "Payment_Method"] = "Invalid Method"

    # Add one exact duplicate.
    dataframe = pd.concat(
        [dataframe, dataframe.iloc[[2]]],
        ignore_index=True,
    )

    # Add one repeated transaction ID with different details.
    conflicting_record = dataframe.iloc[3].copy()
    conflicting_record["Location"] = "Suburban"

    dataframe = pd.concat(
        [
            dataframe,
            pd.DataFrame([conflicting_record]),
        ],
        ignore_index=True,
    )

    return dataframe


def prepare_sample_for_splitting(
    sample_dataset,
    tmp_path,
):
    cleaned = clean_dataset(
        sample_dataset,
        tmp_path,
    )

    groups = cleaned["Transaction_ID"].copy()

    engineered = engineer_features(cleaned)

    splits = split_dataset(
        engineered,
        groups=groups,
    )

    return cleaned, engineered, groups, splits


def test_inspection_generates_reports(
    sample_dataset,
    tmp_path,
):
    report = inspect_dataset(
        sample_dataset,
        tmp_path,
    )

    assert report["rows"] == 42
    assert report["duplicate_rows"] == 1
    assert report["conflicting_transaction_ids"] == 1

    assert (tmp_path / "dataset_profile.json").exists()
    assert (tmp_path / "missing_values.csv").exists()
    assert (tmp_path / "numerical_summary.csv").exists()
    assert (tmp_path / "categorical_summary.csv").exists()


def test_cleaning_removes_exact_duplicates(
    sample_dataset,
    tmp_path,
):
    cleaned = clean_dataset(
        sample_dataset,
        tmp_path,
    )

    assert len(cleaned) == 41
    assert cleaned.duplicated().sum() == 0


def test_cleaning_preserves_conflicting_transaction_ids(
    sample_dataset,
    tmp_path,
):
    cleaned = clean_dataset(
        sample_dataset,
        tmp_path,
    )

    assert cleaned["Transaction_ID"].duplicated().sum() == 1

    assert (
        tmp_path / "conflicting_transaction_records.csv"
    ).exists()


def test_cleaning_handles_invalid_payment_method(
    sample_dataset,
    tmp_path,
):
    cleaned = clean_dataset(
        sample_dataset,
        tmp_path,
    )

    assert cleaned["Payment_Method"].isna().sum() == 1


def test_cleaning_preserves_binary_target(
    sample_dataset,
    tmp_path,
):
    cleaned = clean_dataset(
        sample_dataset,
        tmp_path,
    )

    assert set(cleaned["Fraudulent"].unique()) == {0, 1}


def test_feature_engineering_creates_expected_features(
    sample_dataset,
):
    engineered = engineer_features(sample_dataset)

    expected_features = [
        "Transaction_Hour_Sin",
        "Transaction_Hour_Cos",
        "Transaction_Amount_Log",
        "Account_Age_Log",
    ]

    for feature in expected_features:
        assert feature in engineered.columns


def test_feature_engineering_removes_identifiers(
    sample_dataset,
):
    engineered = engineer_features(sample_dataset)

    assert "Transaction_ID" not in engineered.columns
    assert "User_ID" not in engineered.columns
    assert "Time_of_Transaction" not in engineered.columns


def test_feature_engineering_preserves_target(
    sample_dataset,
):
    engineered = engineer_features(sample_dataset)

    assert "Fraudulent" in engineered.columns


def test_dataset_is_split_into_three_sets(
    sample_dataset,
    tmp_path,
):
    _, _, _, splits = prepare_sample_for_splitting(
        sample_dataset,
        tmp_path,
    )

    assert len(splits["X_train"]) > 0
    assert len(splits["X_validation"]) > 0
    assert len(splits["X_test"]) > 0


def test_split_preserves_both_target_classes(
    sample_dataset,
    tmp_path,
):
    _, _, _, splits = prepare_sample_for_splitting(
        sample_dataset,
        tmp_path,
    )

    for target in [
        splits["y_train"],
        splits["y_validation"],
        splits["y_test"],
    ]:
        assert set(target.unique()) == {0, 1}


def test_transaction_groups_do_not_cross_splits(
    sample_dataset,
    tmp_path,
):
    _, _, _, splits = prepare_sample_for_splitting(
        sample_dataset,
        tmp_path,
    )

    train_groups = splits["train_groups"]
    validation_groups = splits["validation_groups"]
    test_groups = splits["test_groups"]

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_split_rejects_groups_with_conflicting_targets():
    dataframe = pd.DataFrame({
        "feature": list(range(40)),
        "Fraudulent": [0, 1] * 20,
    })

    groups = [f"G{i}" for i in range(40)]

    # Give two records the same group but conflicting labels.
    groups[1] = groups[0]

    with pytest.raises(
        ValueError,
        match="conflicting target labels",
    ):
        split_dataset(
            dataframe,
            groups=groups,
        )


def test_preprocessing_pipeline_handles_missing_values(
    sample_dataset,
    tmp_path,
):
    _, _, _, splits = prepare_sample_for_splitting(
        sample_dataset,
        tmp_path,
    )

    X_train = splits["X_train"]

    numerical_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_validation_processed = preprocessor.transform(
        splits["X_validation"]
    )

    assert X_train_processed.shape[0] == len(X_train)

    assert (
        X_validation_processed.shape[0]
        == len(splits["X_validation"])
    )

    assert X_train_processed.shape[1] > 0

    assert (
        X_validation_processed.shape[1]
        == X_train_processed.shape[1]
    )
