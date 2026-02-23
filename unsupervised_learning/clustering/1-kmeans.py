#!/usr/bin/env python3
"""Performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters
        iterations: maximum number of iterations

    Returns:
        C: numpy.ndarray of shape (k, d) containing centroid means
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    if not isinstance(X, np.ndarray) or X.size == 0 or k <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids with uniform distribution
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    C = np.random.uniform(mins, maxs, (k, d))  # first uniform

    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):
        # Assign points to closest centroid (no loop)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        # Compute new centroids
        new_C = np.zeros((k, d))
        counts = np.zeros(k)

        for idx in range(k):  # loop 1
            points = X[new_clss == idx]
            if len(points) == 0:
                # Reinitialize empty cluster (uniform #2)
                new_C[idx] = np.random.uniform(mins, maxs, d)
            else:
                new_C[idx] = points.mean(axis=0)
            counts[idx] = len(points)

        # Early stopping if centroids didn't move
        if np.allclose(C, new_C):
            break

        C = new_C
        clss = new_clss

    # Final cluster assignment
    clss = np.argmin(np.linalg.norm(X[:, np.newaxis] - C, axis=2), axis=1)
    return C, clss
