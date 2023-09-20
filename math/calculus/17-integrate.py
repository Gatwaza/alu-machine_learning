#!/usr/bin/env python3

def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.
                     Index of the list represents the power of x.
                     Example: if f(x) = x^3 + 3x + 5, poly is equal to [5, 3, 0, 1].
        C (int): Integration constant (default is 0).

    Returns:
        list: List of coefficients representing the integral of the polynomial.
              The returned list is as small as possible.

    Raises:
        ValueError: If poly is not a list of integers or if C is not an integer.

    Example:
        >>> poly_integral([5, 3, 0, 1], 2)
        [0, 5, 1, 0, 0.5]
    """
    if not isinstance(poly, list) or not all(isinstance(coeff, int) for coeff in poly):
        raise ValueError("poly must be a list of integers")
    if not isinstance(C, int):
        raise ValueError("C must be an integer")

    integral_coeffs = [C]
    for i, coeff in enumerate(poly):
        power = i + 1
        integral_coeff = coeff / power if coeff % power == 0 else coeff / power
        integral_coeffs.append(int(integral_coeff))

    return integral_coeffs

# Example usage

