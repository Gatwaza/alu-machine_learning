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
