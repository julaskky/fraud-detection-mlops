import numpy as np
import pandas as pd


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictive features and remove identifier columns.
    """

    dataframe = dataframe.copy()

    # Transaction time is cyclical: 23:00 is close to 00:00.
    dataframe["Transaction_Hour_Sin"] = np.sin(
        2 * np.pi * dataframe["Time_of_Transaction"] / 24
    )

    dataframe["Transaction_Hour_Cos"] = np.cos(
        2 * np.pi * dataframe["Time_of_Transaction"] / 24
    )

    # Reduce the effect of highly skewed numerical distributions.
    dataframe["Transaction_Amount_Log"] = np.log1p(
        dataframe["Transaction_Amount"]
    )

    dataframe["Account_Age_Log"] = np.log1p(
        dataframe["Account_Age"]
    )

    # The original hour is represented by its cyclical features.
    dataframe = dataframe.drop(
        columns=["Time_of_Transaction"]
    )

    # Identifiers must not be used as predictive features.
    dataframe = dataframe.drop(
        columns=["Transaction_ID", "User_ID"],
        errors="ignore",
    )

    return dataframe