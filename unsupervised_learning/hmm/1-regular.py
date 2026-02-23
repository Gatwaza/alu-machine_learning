#!/usr/bin/env python3
"""Module for computing steady state probabilities of a regular
Markov chain."""
import numpy as np


def regular(P):
    """Determine the steady state probabilities of a regular Markov chain.

    Args:
        P: numpy.ndarray of shape (n, n), the transition matrix

    Returns:
        numpy.ndarray of shape (1, n) with steady state probabilities,
        or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not np.isclose(P.sum(axis=1), 1).all():
        return None
    if np.any(P <= 0):
        return None

    n = P.shape[0]
    Pt = P.T
    A = Pt - np.eye(n)
    A[-1] = 1
    b = np.zeros(n)
    b[-1] = 1
    try:
        steady = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return steady.reshape(1, n)
