#!/usr/bin/env python3
"""Module for finding the best number of clusters using BIC for a GMM."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, minimum number of clusters (inclusive)
        kmax: positive integer, maximum number of clusters (inclusive)
        iterations: positive integer, maximum iterations for EM algorithm
        tol: non-negative float, tolerance for EM algorithm
        verbose: boolean, if True EM algorithm prints information

    Returns:
        best_k: best value for k based on BIC
        best_result: tuple (pi, m, S) for the best k
        l: numpy.ndarray of shape (kmax - kmin + 1,) with log likelihoods
        b: numpy.ndarray of shape (kmax - kmin + 1,) with BIC values
        or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmax < kmin:
        return None, None, None, None

    results = []
    log_likelihoods = []
    bic_values = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose)
        if pi is None:
            return None, None, None, None

        results.append((pi, m, S))
        log_likelihoods.append(log_l)

        # p: number of free parameters in a k-component GMM
        # - k-1 priors (pi sums to 1)
        # - k * d means
        # - k * d*(d+1)/2 covariance parameters (symmetric matrix)
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)
        bic = p * np.log(n) - 2 * log_l
        bic_values.append(bic)

    log_likelihoods = np.array(log_likelihoods)
    bic_values = np.array(bic_values)

    best_idx = np.argmin(bic_values)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, log_likelihoods, bic_values
