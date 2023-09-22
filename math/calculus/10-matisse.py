#!/usr/bin/env python3
"""
Function that calculates derivatives of polynomial.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.

    Returns:
        list: List of coefficients representing
              the derivative of the polynomial.
              If the derivative is 0, returns [0].
              If poly is not valid, returns None.
    """
    if not isinstance(poly, list):
        return None

    n = len(poly)
    if n == 0:
        return None

    derivative = [i * poly[i] for i in range(1, n)]
    return derivative if any(derivative) else [0]
