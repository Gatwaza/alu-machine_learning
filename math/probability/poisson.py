#!/usr/bin/env python3
"""
class Poisson that represents a poisson distribution
"""


class Poisson:
    """
    Class representing a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize a Poisson instance.

        Args:
            data (list, optional): List of data points. Defaults to None.
            lambtha (float, optional):
            Expected number of occurrences. Defaults to 1.0.

        Raises:
            ValueError: If lambtha is not a positive value or equals to 0.
            TypeError: If data is not a list or
            contains less than two data points.
        """
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")

        if data is None:
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = sum(data) / len(data)