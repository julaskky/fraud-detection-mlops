
from sklearn.model_selection import StratifiedGroupKFold

import numpy as np
import pandas as pd


def split_dataset(
    dataframe: pd.DataFrame,
    target_column: str = "Fraudulent",
    groups=None,
    random_state: int = 42,
) -> dict:
    """
    Split data into training, validation and test sets.

    Uses stratified group splitting to prevent records
    belonging to the same transaction from crossing
    dataset boundaries.

    Approximate distribution:
        Training:   70%
        Validation: 15%
        Testing:    15%
    """

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if y.nunique() != 2:
        raise ValueError(
            "This pipeline expects two target classes."
        )

    # If groups are not supplied, each record is treated
    # as its own group. Production preparation supplies
    # Transaction_ID explicitly.
    if groups is None:
        groups_array = np.arange(len(dataframe))
    else:
        groups_array = np.asarray(groups)

    if len(groups_array) != len(dataframe):
        raise ValueError(
            "Groups must contain one identifier per record."
        )

    if pd.isna(groups_array).any():
        raise ValueError(
            "Group identifiers cannot contain missing values."
        )

    unique_groups = pd.Series(groups_array).nunique()

    if unique_groups < 20:
        raise ValueError(
            "At least 20 unique groups are required."
        )

    # Every group must have a consistent target label.
    group_labels = pd.DataFrame({
        "group": groups_array,
        "target": y.to_numpy(),
    })

    inconsistent_groups = (
        group_labels.groupby("group")["target"]
        .nunique()
    )

    if (inconsistent_groups > 1).any():
        raise ValueError(
            "Some groups have conflicting target labels."
        )

    group_class_counts = (
        group_labels.drop_duplicates("group")["target"]
        .value_counts()
    )

    if (group_class_counts < 20).any():
        raise ValueError(
            "Each target class must contain at least "
            "20 unique groups for this splitting strategy."
        )

    splitter = StratifiedGroupKFold(
        n_splits=20,
        shuffle=True,
        random_state=random_state,
    )

    folds = list(
        splitter.split(X, y, groups_array)
    )

    # Each fold represents approximately 5% of the data.
    test_indices = np.concatenate([
        folds[i][1] for i in range(3)
    ])

    validation_indices = np.concatenate([
        folds[i][1] for i in range(3, 6)
    ])

    train_indices = np.concatenate([
        folds[i][1] for i in range(6, 20)
    ])

    # Confirm every record is assigned exactly once.
    all_indices = np.concatenate([
        train_indices,
        validation_indices,
        test_indices,
    ])

    if (
        len(all_indices) != len(dataframe)
        or len(np.unique(all_indices)) != len(dataframe)
    ):
        raise RuntimeError(
            "Dataset splitting did not assign every record "
            "exactly once."
        )

    # Verify that groups do not cross dataset boundaries.
    train_groups = set(groups_array[train_indices])
    validation_groups = set(groups_array[validation_indices])
    test_groups = set(groups_array[test_indices])

    if train_groups & validation_groups:
        raise RuntimeError(
            "Transaction leakage detected between training "
            "and validation datasets."
        )

    if train_groups & test_groups:
        raise RuntimeError(
            "Transaction leakage detected between training "
            "and test datasets."
        )

    if validation_groups & test_groups:
        raise RuntimeError(
            "Transaction leakage detected between validation "
            "and test datasets."
        )

    return {
        "X_train": X.iloc[train_indices],
        "X_validation": X.iloc[validation_indices],
        "X_test": X.iloc[test_indices],
        "y_train": y.iloc[train_indices],
        "y_validation": y.iloc[validation_indices],
        "y_test": y.iloc[test_indices],
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "test_groups": test_groups,
    }
