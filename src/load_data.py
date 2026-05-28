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

def export_data(df, path: str) -> pd.DataFrame:
    
    """
    Export dataset to CSV file.

    Args:
        path (str): Path to CSV file.
    """
    df.to_csv(path + '/dataset_clean.csv', index=False)