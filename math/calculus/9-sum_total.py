#!/usr/bin/env python3
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
    if isinstance(n, int) and n > 0:
        return (n * (n + 1) * (2*n + 1)) // 6
    else:
        raise ValueError("n must be a positive integer.")
