"""
Unsupervised learning model.
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd


def run_kmeans(scaled_features) -> None:
    """
    Apply KMeans clustering.

    Args:
        scaled_features: Scaled numerical data.
    """

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(scaled_features)

    score = silhouette_score(scaled_features, clusters)

    print("\n===== UNSUPERVISED MODEL =====")
    print(f"Silhouette Score: {score:.4f}")