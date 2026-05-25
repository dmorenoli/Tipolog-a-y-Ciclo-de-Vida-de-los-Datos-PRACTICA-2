"""
Functions for loading the dataset.
"""

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(path)
    return df