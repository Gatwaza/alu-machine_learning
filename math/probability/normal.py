#!/usr/bin/env python3
"""
Create a class Normal that represents a normal distribution
"""


class Normal:
    """
    Class representing a normal distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize a Normal instance.

        Args:
            data (list, optional): List of data points. Defaults to None.
            mean (float, optional): Mean of the distribution. Defaults to 0.0.
            stddev (float, optional): Standard deviation of
            the distribution. Defaults to 1.0.

        Raises:
            ValueError: If stddev is not a positive value or equals to 0.
            TypeError: If data is not a list or contains
            less than two data points.
        """
        if stddev <= 0:
            raise ValueError("stddev must be a positive value")

        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
            self.stddev = (sum((x - self.mean)
                               ** 2 for x in data) / len(data)) ** 0.5

    def z_score(self, x):
        """
        Calculate the z-score of a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value of a given z-score.

        Args:
            z (float): The z-score.

        Returns:
            float: The x-value of z.
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculate the value of the Probability Density Function (PDF)
        for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The PDF value for x.
        """
        constant = 1 / (self.stddev * (2 * 3.1415926536) ** 0.5)
        exponent = -(x - self.mean) ** 2 / (2 * self.stddev ** 2)
        return constant * (2.7182818285 ** exponent)

    def cdf(self, x):
        """
        Calculate the value of the Cumulative Distribution Function (CDF)
        for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The CDF value for x.
        """
        constant = 0.5 * (
            1 + self.er((x - self.mean) / (self.stddev * (2 ** 0.5)))
        )
        return constant

    def er(self, x):
        """
        Approximate the error function using a given polynomial.

        Args:
            x (float): The x-value.

        Returns:
            float: The approximate value of the error function.
        """
        return (
            2 / (3.1415926536 ** 0.5) *
            (x - (x ** 3) / 3 + (x ** 5) / 10 - (x ** 7) / 42 + (x ** 9) / 216)
        )
