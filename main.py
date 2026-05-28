"""
Main execution file.
"""

from src.load_data import load_dataset, export_data
from src.preprocessing import (
    replace_invalid_zeros,
    impute_missing_values,
    create_categorical_variables,
    scale_features
)

from src.visualization import (
    generate_boxplots,
    correlation_heatmap
)

from src.supervised_model import run_logistic_regression

from src.unsupervised_model import run_kmeans

from src.hypothesis_testing import glucose_hypothesis_test

from src.utils import dataset_summary


def main():
    """
    Main program execution.
    """

    # Load data
    df = load_dataset("data/diabetes.csv")

    # Summary before cleaning
    dataset_summary(df)

    # Cleaning
    df = replace_invalid_zeros(df)

    df = impute_missing_values(df)

    df = create_categorical_variables(df)

    # Export
    
    export_data(df,"outputs/reports/")
    
    # Visualizations
    generate_boxplots(df)

    correlation_heatmap(df)

    # Scaling
    scaled_features, scaler = scale_features(df)

    # Models
    run_logistic_regression(df)

    run_kmeans(scaled_features)

    # Hypothesis testing
    glucose_hypothesis_test(df)


if __name__ == "__main__":
    main()