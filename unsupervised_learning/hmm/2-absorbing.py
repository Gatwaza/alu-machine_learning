#!/usr/bin/env python3
"""Module for determining if a Markov chain is absorbing."""
import numpy as np


def absorbing(P):
    """Determine if a Markov chain is absorbing.

    Args:
        P: numpy.ndarray of shape (n, n), the transition matrix

    Returns:
        True if the chain is absorbing, False on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if not np.isclose(P.sum(axis=1), 1).all():
        return False

    n = P.shape[0]
    absorbing_states = np.where(np.isclose(np.diag(P), 1))[0]

    if len(absorbing_states) == 0:
        return False

    reachable = set(absorbing_states)
    prev_size = 0

    while len(reachable) != prev_size:
        prev_size = len(reachable)
        for i in range(n):
            if i not in reachable:
                if any(P[i, j] > 0 for j in reachable):
                    reachable.add(i)

    return len(reachable) == n
