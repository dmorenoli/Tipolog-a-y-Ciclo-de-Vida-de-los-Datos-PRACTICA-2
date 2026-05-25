"""
Utility functions.
"""

import pandas as pd


def dataset_summary(df: pd.DataFrame) -> None:
    """
    Print dataset summary.

    Args:
        df (pd.DataFrame): Input dataframe.
    """

    print("\n===== DATASET INFO =====")

    print(df.info())

    print("\n===== DESCRIPTIVE STATISTICS =====")

    print(df.describe())

    print("\n===== MISSING VALUES =====")

    print(df.isnull().sum())