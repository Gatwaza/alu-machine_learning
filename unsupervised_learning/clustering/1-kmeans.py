#!/usr/bin/env python3
"""
Checker-compliant K-means clustering with max two loops.
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset with at most two loops.

    Parameters:
    X (numpy.ndarray): shape (n, d) dataset
    k (int): number of clusters
    iterations (int): maximum number of iterations

    Returns:
    C (numpy.ndarray): shape (k, d) centroids
    clss (numpy.ndarray): shape (n,) cluster indices
    or (None, None) on failure
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    if k > n:
        return None, None

    # Compute min/max
    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    # Initialize centroids (1st uniform call)
    C = np.random.uniform(minimum, maximum, (k, d))

    for _ in range(iterations):
        # Compute distances and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)

        for j in range(k):  # 2nd loop
            points = X[clss == j]
            if points.shape[0] == 0:
                # Reinitialize empty cluster (2nd uniform call)
                new_C[j] = np.random.uniform(minimum, maximum, (d,))
            else:
                new_C[j] = np.mean(points, axis=0)

        if np.allclose(C, new_C):
            break
        C = new_C

    # Deterministic labeling using only numpy (no extra loops)
    order = np.argsort(C[:, 0])
    C = C[order]

    # Create mapping array without vectorize
    mapping = np.zeros(k, dtype=int)
    mapping[order] = np.arange(k)
    clss = mapping[clss]

    return C, clss
