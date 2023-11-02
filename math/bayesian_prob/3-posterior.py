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
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate binomial probabilities
    likelihoods = np.array([
        np.math.comb(n, x) * (p**x) * ((1-p)**(n-x)) for p in P
    ])

    return likelihoods


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining
    data with various hypothetical probabilities.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy array containing
        the various hypothetical probabilities
            of developing severe side effects.
        Pr (numpy.ndarray): A 1D numpy array containing the prior beliefs of P.

    Returns:
        numpy.ndarray: A 1D numpy array containing
        the intersection of obtaining x and n
            with each probability in P, respectively.

    Raises:
        ValueError: If n is not a positive integer,
        x is not a non-negative integer,
            x is greater than n, or any value in P or
            Pr is not in the range [0, 1].
        TypeError: If P is not a 1D numpy.ndarray, or
        Pr is not a numpy.ndarray with
            the same shape as P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError(f"All values in {P} must be in the range [0, 1]")
    if not all(0 <= pr <= 1 for pr in Pr):
        raise ValueError(f"All values in {Pr} must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the intersection
    intersection_values = np.array([
        likelihood(x, n, np.array([p]))[0] * pr for p, pr in zip(P, Pr)
        ])

    return intersection_values


def marginal(x, n, P, Pr):
    """
    Calculate the marginal probability of obtaining the data.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy array
        containing the various hypothetical probabilities
            of patients developing severe side effects.
        Pr (numpy.ndarray): A 1D numpy array
        containing the prior beliefs about P.

    Returns:
        float: The marginal probability of obtaining x and n.

    Raises:
        ValueError: If n is not a positive integer,
        x is not a non-negative integer,
            x is greater than n, or any value in P or
            Pr is not in the range [0, 1].
        TypeError: If P is not a 1D numpy.ndarray, or
        Pr is not a numpy.ndarray with
            the same shape as P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError(f"All values in {P} must be in the range [0, 1]")
    if not all(0 <= pr <= 1 for pr in Pr):
        raise ValueError(f"All values in {Pr} must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the marginal probability
    marginal_prob = np.sum(likelihood(x, n, P) * Pr)

    return marginal_prob


def posterior(x, n, P, Pr):
    """
    Calculate the posterior probability for
    the various hypothetical probabilities
    of developing severe side effects given the data.

    Args:
        x (int): The number of patients that
        develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy array containing
        the various hypothetical probabilities
            of developing severe side effects.
        Pr (numpy.ndarray): A 1D numpy array
        containing the prior beliefs of P.

    Returns:
        numpy.ndarray: A 1D numpy array containing
        the posterior probability of each probability in P
            given x and n, respectively.

    Raises:
        ValueError: If n is not a positive integer,
        x is not a non-negative integer,
            x is greater than n, or any value in P
            or Pr is not in the range [0, 1].
        TypeError: If P is not a 1D numpy.ndarray,
        or Pr is not a numpy.ndarray with
            the same shape as P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
                "x must be an integer that is greater than or equal to 0"
            )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1 or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not all(0 <= p <= 1 for p in P):
        raise ValueError(f"All values in {P} must be in the range [0, 1]")
    if not all(0 <= pr <= 1 for pr in Pr):
        raise ValueError(f"All values in {Pr} must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the posterior probability
    likelihood_values = likelihood(x, n, P)
    posterior_values = (
        likelihood_values * Pr) / np.sum(likelihood_values * Pr
                                         )

    return posterior_values
