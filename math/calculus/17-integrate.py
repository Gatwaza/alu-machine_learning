#!/usr/bin/env python3

def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.
                     The index of the list represents the power of x that the coefficient belongs to.
        C (int, optional): Integration constant. Default is 0.

    Returns:
        list: A new list of coefficients representing the integral of the polynomial.

    Raises:
        TypeError: If poly is not a list of integers or if C is not an integer.

    Example:
        >>> poly_integral([5, 3, 0, 1], 2)
        [2, 0, 1.5, 0, 0.25]
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        raise TypeError("poly must be a list of integers or floats")

    if not isinstance(C, (int, float)):
        raise TypeError("C must be an integer or float")

    result = [C]
    for i, coef in enumerate(poly):
        exponent = i + 1
        new_coef = coef / exponent if exponent > 0 else 0
        result.append(new_coef)

    # Remove trailing zeros
    while result and result[-1] == 0:
        result.pop()

    return result

if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    C = 0
    print(poly_integral(poly, C))
