#!/usr/bin/env python3
"""Module for K-means clustering algorithm."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer for maximum iterations

    Returns:
        C: numpy.ndarray of shape (k, d) with centroid means
        clss: numpy.ndarray of shape (n,) with cluster indices
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        dists = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dists, axis=1)
        C_new = np.zeros((k, d))
        for j in range(k):
            pts = X[clss == j]
            if len(pts) == 0:
                C_new[j] = np.random.uniform(low, high)
            else:
                C_new[j] = pts.mean(axis=0)
        if np.allclose(C, C_new):
            return C_new, clss
        C = C_new

    dists = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(dists, axis=1)
    return C, clss
