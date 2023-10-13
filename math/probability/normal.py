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
        Calculate the value of the PDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The PDF value for x.
        """
        pi_approx = 3.1415926536
        e_approx = 2.7182818285
        part1 = 1 / (self.stddev * (e_approx ** 0.5 * pi_approx))
        exponent = -(x - self.mean) ** 2 / (2 * self.stddev ** 2)
        part2 = (e_approx ** exponent)

        return part1 * part2

    def cdf(self, x):
        """
        Calculate the value of the CDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The CDF value for x.
        """
        z = self.z_score(x)
        return (1 + self.error_function(z / (2 ** 0.5))) / 2

    def error_function(self, x):
        """
        Approximation of the error function.

        Args:
            x (float): The value for which to approximate the error function.

        Returns:
            float: The approximate value of the error function.
        """
        t = 1 / (1 + 0.5 * abs(x))
        result = t * 2.7182818285 ** (-x * x - 1.26551223 + t *
                                      (1.00002368 + t *
                                       (0.37409196 + t *
                                        (0.09678418 +
                                         t * (-0.18628806 + t *
                                              (0.27886807 + t *
                                               (-1.13520398 + t *
                                                (1.48851587 + t *
                                                 (-0.82215223 +
                                                  t * 0.17087277)))))))))
        return 1 - result if x >= 0 else result
