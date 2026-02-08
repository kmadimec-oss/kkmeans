# -*- coding: utf-8 -*-
"""Untitled7.ipynb



Original file is located at
    https://colab.research.google.com/drive/1ljwP6svXKUhDxyL2msuZTG3QWQuyCjAi
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


class KMeansFromScratch:
    """
    K-Means clustering implemented from scratch using NumPy only.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids_kmeans_pp(self, X):
        """
        Initialize centroids using the K-Means++ algorithm.
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # Choose first centroid randomly
        centroids[0] = X[np.random.choice(n_samples)]

        # Choose remaining centroids
        for k in range(1, self.n_clusters):
            distances = np.min(
                np.linalg.norm(X[:, np.newaxis] - centroids[:k], axis=2) ** 2,
                axis=1
            )
            probabilities = distances / np.sum(distances)
            centroids[k] = X[np.random.choice(n_samples, p=probabilities)]

        return centroids

    def _assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """
        Update centroids as the mean of assigned points.
        """
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
            for k in range(self.n_clusters)
        ])
        return new_centroids

    def _compute_inertia(self, X, labels):
        """
        Compute Sum of Squared Errors (SSE / inertia).
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        return inertia

    def fit(self, X):
        """
        Fit the K-Means model to the data.
        """
        self.centroids = self._initialize_centroids_kmeans_pp(X)

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)

            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                break

        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)
        return self


def main():
    # Load dataset
    data = load_iris()
    X = data.data

    print("K-Means Clustering Comparison (Custom vs. sklearn)")
    print("-" * 60)
    print(f"{'K':<5}{'Custom SSE':<20}{'sklearn SSE':<20}")
    print("-" * 60)

    results = []

    for k in range(2, 7):
        # Custom K-Means
        custom_kmeans = KMeansFromScratch(n_clusters=k)
        custom_kmeans.fit(X)

        # sklearn K-Means
        sklearn_kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=42
        )
        sklearn_kmeans.fit(X)

        custom_sse = custom_kmeans.inertia_
        sklearn_sse = sklearn_kmeans.inertia_

        results.append((k, custom_sse, sklearn_sse))

        print(f"{k:<5}{custom_sse:<20.4f}{sklearn_sse:<20.4f}")

    print("\nElbow Method Visualization Description:")
    print(
        "A line plot would show K values (2–6) on the x-axis and SSE on the y-axis. "
        "Both curves (custom and sklearn) would decrease as K increases, with a "
        "noticeable 'elbow' around K=3, indicating diminishing returns in SSE reduction."
    )


if __name__ == "__main__":
    main()
