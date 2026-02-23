#!/usr/bin/env python3
"""Module for the Baum-Welch algorithm for a hidden Markov model."""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Perform the Baum-Welch algorithm for a hidden Markov model.

    Args:
        Observations: numpy.ndarray of shape (T,) with observation indices
        Transition: numpy.ndarray of shape (M, M) with transition probabilities
        Emission: numpy.ndarray of shape (M, N) with emission probabilities
        Initial: numpy.ndarray of shape (M, 1) with initial state probabilities
        iterations: number of EM iterations to perform

    Returns:
        Transition: converged transition matrix
        Emission: converged emission matrix
        or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    M, N = Emission.shape
    T = Observations.shape[0]

    if Transition.shape != (M, M) or Initial.shape != (M, 1):
        return None, None

    for _ in range(iterations):
        F = np.zeros((M, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            F[:, t] = (np.matmul(F[:, t - 1], Transition) *
                       Emission[:, Observations[t]])

        B = np.zeros((M, T))
        B[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            B[:, t] = np.matmul(
                Transition, Emission[:, Observations[t + 1]] * B[:, t + 1])

        a_t = F[:, :T - 1]
        b_t1 = B[:, 1:T]
        e_t1 = Emission[:, Observations[1:T]]

        xi = (a_t[:, np.newaxis, :] *
              Transition[:, :, np.newaxis] *
              (e_t1 * b_t1)[np.newaxis, :, :])
        xi /= xi.sum(axis=(0, 1), keepdims=True)

        gamma = xi.sum(axis=1)
        last = xi[:, :, T - 2].sum(axis=1, keepdims=True)
        gamma_full = np.concatenate((gamma, last), axis=1)

        Transition = xi.sum(axis=2) / gamma.sum(axis=1, keepdims=True)

        for k in range(N):
            Emission[:, k] = (gamma_full[:, Observations == k].sum(axis=1) /
                              gamma_full.sum(axis=1))

    return Transition, Emission
