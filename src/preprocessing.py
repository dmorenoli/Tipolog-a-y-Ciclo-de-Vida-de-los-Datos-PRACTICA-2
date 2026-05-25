"""
Data preprocessing and cleaning functions.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace invalid zero values with NaN.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Modified dataframe.
    """

    columns_with_invalid_zeros = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    df[columns_with_invalid_zeros] = (
        df[columns_with_invalid_zeros].replace(0, np.nan)
    )

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using median strategy.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with imputed values.
    """

    imputer = SimpleImputer(strategy="median")

    numeric_columns = df.select_dtypes(include=np.number).columns

    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    return df


def create_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create categorical variables from numerical variables.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Modified dataframe.
    """

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 50, 100],
        labels=["Young", "Adult", "Senior"]
    )

    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )
    
    df["Glucose_Category"] = pd.cut(
        df["Glucose"],
        bins=[0, 69, 99, 125, 200],
        labels=["Hypoglycemia", "Normal", "Prediabetes", "Diabetes"]
    )

    return df


def scale_features(df: pd.DataFrame):
    """
    Scale numerical features using StandardScaler.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        tuple:
            - scaled features
            - scaler object
    """

    scaler = StandardScaler()

    features = df.drop(columns=["Outcome", "AgeGroup", "BMI_Category", "Glucose_Category"])

    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler