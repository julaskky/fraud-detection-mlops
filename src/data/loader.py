from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "Transaction_ID",
    "User_ID",
    "Transaction_Amount",
    "Transaction_Type",
    "Time_of_Transaction",
    "Device_Used",
    "Location",
    "Previous_Fraudulent_Transactions",
    "Account_Age",
    "Number_of_Transactions_Last_24H",
    "Payment_Method",
    "Fraudulent",
]


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load the fraud detection dataset from a CSV file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError("Dataset must be a CSV file.")

    dataframe = pd.read_csv(file_path)

    return dataframe


def validate_columns(dataframe: pd.DataFrame) -> None:
    """
    Validate that the dataset contains the expected columns.
    """

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_target(dataframe: pd.DataFrame) -> None:
    """
    Validate the Fraudulent target column.
    """

    if "Fraudulent" not in dataframe.columns:
        raise ValueError(
            "Target column 'Fraudulent' is missing."
        )

    target_values = set(
        dataframe["Fraudulent"].dropna().unique()
    )

    invalid_values = target_values - {0, 1}

    if invalid_values:
        raise ValueError(
            f"Invalid target values found: {invalid_values}"
        )


def validate_time(dataframe: pd.DataFrame) -> None:
    """
    Validate that transaction time is within the 24-hour clock.
    """

    if "Time_of_Transaction" not in dataframe.columns:
        raise ValueError(
            "Time_of_Transaction column is missing."
        )

    valid_times = dataframe[
        "Time_of_Transaction"
    ].dropna()

    invalid_times = valid_times[
        (valid_times < 0) | (valid_times > 23)
    ]

    if not invalid_times.empty:
        raise ValueError(
            "Time_of_Transaction contains values outside 0-23."
        )


def validate_account_age(dataframe: pd.DataFrame) -> None:
    """
    Validate that account age is expressed as non-negative days.
    """

    if "Account_Age" not in dataframe.columns:
        raise ValueError(
            "Account_Age column is missing."
        )

    account_age = dataframe[
        "Account_Age"
    ].dropna()

    invalid_values = account_age[
        account_age < 0
    ]

    if not invalid_values.empty:
        raise ValueError(
            "Account_Age contains negative values."
        )


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """
    Run all Phase 1 dataset validation checks.
    """

    validate_columns(dataframe)
    validate_target(dataframe)
    validate_time(dataframe)
    validate_account_age(dataframe)