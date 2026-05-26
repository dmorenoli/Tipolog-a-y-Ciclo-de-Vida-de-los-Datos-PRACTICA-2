"""
Supervised learning model.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd


def run_logistic_regression(df: pd.DataFrame) -> None:
    """
    Train and evaluate logistic regression model.

    Args:
        df (pd.DataFrame): Input dataframe.
    """

    X = df.drop(columns=["Outcome", "AgeGroup", "BMI_Category", "Glucose_Category"])

    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions)

    print("\n===== SUPERVISED MODEL =====")
    print(report)