"""
Hypothesis testing functions.
"""

from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import mannwhitneyu
import pandas as pd


def glucose_hypothesis_test(df: pd.DataFrame) -> None:
    """
    Compare glucose levels between diabetic and non-diabetic patients.

    Args:
        df (pd.DataFrame): Input dataframe.
    """

    diabetic = df[df["Outcome"] == 1]["Glucose"]

    non_diabetic = df[df["Outcome"] == 0]["Glucose"]

    print("\n===== HYPOTHESIS TESTING =====")

    # Normality test
    stat1, p1 = shapiro(diabetic)

    stat2, p2 = shapiro(non_diabetic)

    print(f"Shapiro diabetic p-value: {p1:.4f}")
    print(f"Shapiro non-diabetic p-value: {p2:.4f}")

    # Homogeneity of variances
    stat3, p3 = levene(diabetic, non_diabetic)

    print(f"Levene p-value: {p3:.4f}")

    # Non-parametric test
    stat4, p4 = mannwhitneyu(diabetic, non_diabetic)

    print(f"Mann-Whitney U p-value: {p4:.4f}")

    if p4 < 0.05:
        print("There is a statistically significant difference.")
    else:
        print("There is no statistically significant difference.")