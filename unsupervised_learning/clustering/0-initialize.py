#!/usr/bin/env python3
"""
Module for initializing K-means cluster centroids.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters:
    X (numpy.ndarray): shape (n, d) dataset
    k (int): number of clusters

    Returns:
    numpy.ndarray: shape (k, d) initialized centroids,
    or None on failure
    """
    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(k, int) or
            k <= 0):
        return None

    n, d = X.shape
    if k > n:
        return None

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    centroids = np.random.uniform(
        low=minimum,
        high=maximum,
        size=(k, d)
    )

    return centroids
