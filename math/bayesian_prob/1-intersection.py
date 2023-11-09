#!/usr/bin/env python3
"""
Function that calculate likelihood of probability.
"""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining
    data with the various hypothetical probabilities.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy array
            containing the various hypothetical probabilities
            of developing severe side effects.
        Pr (numpy.ndarray): A 1D numpy array containing
            the prior beliefs of P.

    Returns:
        numpy.ndarray: A 1D numpy array containing
            the intersection of obtaining the data, x and n,
            with each probability in P, respectively.

    Raises:
        ValueError: If n is not a positive integer,
            x is not a non-negative integer,
            x is greater than n, any value in P or Pr
            is not in the range [0, 1], or if Pr does not
            sum to 1.
        TypeError: If P or Pr is not a 1D numpy.ndarray,
            or if Pr does not have the same shape as P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
            )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError(f"All values in P must be in the range [0, 1]")
    if not all(0 <= pr <= 1 for pr in Pr):
        raise ValueError(f"All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate binomial probabilities
    binomial_coeff = np.math.factorial(n) / (np.math.factorial(x) *
                                             np.math.factorial(n - x))
    likelihoods = np.array([
        binomial_coeff * (p**x) * ((1-p)**(n-x)) for p in P
    ])
    # Calculate the intersection
    intersection_probabilities = likelihoods * Pr

    return intersection_probabilities
