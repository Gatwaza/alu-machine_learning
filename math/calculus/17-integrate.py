#!/usr/bin/env python3
"""
Function that calculates integral of polynomial.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.
                    The index of the list represents. 
                    the power of x that the coefficient belongs to.
        C (int, optional): Integration constant. Default is 0.

    Returns:
        list: A new list of coefficients representing, 
            the integral of the polynomial.

    Raises:
        TypeError: If poly is not a list or if any coefficient is not 
        an integer or float, or if C is not an integer.

    Example:
        >>> poly_integral([5, 3, 0, 1], 2)
        [2, 0, 1.5, 0, 0.25]
    """
    if not isinstance(poly, list) or \
            not all(isinstance(coef, (int, float)) for coef in poly) or \
            not isinstance(C, (int, float)):
        return None

    if len(poly) == 1 and poly[0] == 0:
        return [C]

    integral_coeffs = [C]
    for i, coef in enumerate(poly, start=1):
        new_coef = coef / i
        if new_coef.is_integer():
            new_coef = int(new_coef)
        integral_coeffs.append(new_coef)

    return integral_coeffs
