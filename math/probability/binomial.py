#!/usr/bin/env python3
"""
Create a class Binomial that represents a binomial distribution
"""


class Binomial:
    """
    Class representing a binomial distribution.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize a Binomial instance.

        Args:
            data (list, optional): List of data points. Defaults to None.
            n (int, optional): Number of Bernoulli trials. Defaults to 1.
            p (float, optional): Probability of success. Defaults to 0.5.

        Raises:
            ValueError: If n is not a positive integer or p is not in the
                        range (0, 1).
            TypeError: If data is not a list or contains less than two
                       data points.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if not (0 < p < 1):
            raise ValueError("p must be greater than 0 and less than 1")

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            successes = sum(1 for d in data if d == 1)
            trials = len(data)
            if trials == 0:
                p = 0.0
                n = 1
            else:
                p = successes / trials
                n = round(trials / p)
                p = successes / n

        self.n = n
        self.p = float(p)
