#!/usr/bin/env python3

def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.
        C (int, optional): Integration constant. Default is 0.

    Returns:
        list: List of coefficients representing the integral of the polynomial.

    Example:
        If f(x) = x^3 + 3x + 5, poly is equal to [5, 3, 0, 1]

    Note:
        - If a coefficient is a whole number, 
        it should be represented as an integer.
        - If poly or C are not valid, return None.
        - The returned list should be as small as possible.
    """
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly) or not isinstance(C, int):
        return None

    result = [C]
    for i, coeff in enumerate(poly):
        if not isinstance(coeff, int):
            coeff = round(coeff)
        integral_coeff = coeff / (i + 1)
        result.append(integral_coeff)

    while result[-1] == 0 and len(result) > 1:
        result.pop()

    return result


# Example usage
poly = [5, 3, 0, 1]
C = 0
integral_poly = poly_integral(poly, C)
print(integral_poly)
