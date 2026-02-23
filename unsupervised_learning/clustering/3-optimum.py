#!/usr/bin/env python3
"""Module for finding the optimum number of clusters by variance."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, minimum number of clusters (inclusive)
        kmax: positive integer, maximum number of clusters (inclusive)
        iterations: positive integer, maximum iterations for K-means

    Returns:
        results: list of K-means outputs for each cluster size
        d_vars: list of variance differences from smallest cluster size
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmax <= kmin or kmax > n:
        return None, None

    results = []
    vars_ = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))
        vars_.append(variance(X, C))

    base = vars_[0]
    d_vars = [base - v for v in vars_]

    return results, d_vars
