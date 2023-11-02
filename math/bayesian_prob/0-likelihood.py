#!/usr/bin/env python3
"""
Function that calculate likelihood of probability.
"""


import numpy as np


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining
    data given various hypothetical probabilities.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy array
        containing the various hypothetical probabilities
                           of developing severe side effects.

    Returns:
        numpy.ndarray: A 1D numpy array containing
        the likelihood of obtaining the data, x and n,
                       for each probability in P, respectively.

    Raises:
        ValueError: If n is not a positive integer,
         x is not a non-negative integer,
                    x is greater than n, or any
                    value in P is not in the range [0, 1].
        TypeError: If P is not a 1D numpy.ndarray.

    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError
    ("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate binomial probabilities
    likelihoods = np.array
    ([np.math.comb(n, x) * (p**x) * ((1-p)**(n-x)) for p in P])

    return likelihoods
