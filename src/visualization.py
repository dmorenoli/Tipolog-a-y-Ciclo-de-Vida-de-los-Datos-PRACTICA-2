"""
Visualization functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_boxplots(df: pd.DataFrame) -> None:
    """
    Generate boxplots for numerical variables.

    Args:
        df (pd.DataFrame): Input dataframe.
    """

    numerical_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "Age"
    ]

    for column in numerical_columns:

        plt.figure(figsize=(8, 4))

        sns.boxplot(x=df[column])

        plt.title(f"Boxplot of {column}")

        plt.savefig(f"outputs/figures/{column}_boxplot.png")

        plt.close()


def correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Generate correlation heatmap.

    Args:
        df (pd.DataFrame): Input dataframe.
    """

    plt.figure(figsize=(10, 8))

    sns.heatmap(df.corr(numeric_only=True),
                annot=True,
                cmap="coolwarm")

    plt.title("Correlation Matrix")

    plt.savefig("outputs/figures/correlation_heatmap.png")

    plt.close()