#!/usr/bin/env python3
"""
Function that calculates summation of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares of the first n natural numbers.

    Args:
        n (int): The stopping condition.

    Returns:
        int: The integer value of the sum.

    Raises:
        TypeError: If n is not an integer.
        ValueError: If n is not a positive integer.

    Example:
        >>> summation_i_squared(3)
        14
        >>> summation_i_squared(5)
        55
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n*(n+1)*(2*n+1)//6