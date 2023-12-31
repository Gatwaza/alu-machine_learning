#!/usr/bin/env python3
"""
Class representing an exponential distribution.
"""


class Exponential:
    """
    Class representing an exponential distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize an Exponential instance.

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

            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Calculate the value of the PDF for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The PDF value for x.
        """
        if x < 0:
            return 0
        else:
            return self.lambtha * 2.7182818285 ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Calculate the value of the CDF for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The CDF value for x.
        """
        if x < 0:
            return 0
        else:
            return 1 - 2.7182818285 ** (-self.lambtha * x)
