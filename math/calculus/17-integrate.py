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
        - If a coefficient is a whole number, it should be represented as an integer.
        - If poly or C are not valid, return None.
        - The returned list should be as small as possible.
    """
    if not isinstance(poly, list) or not all(isinstance(coeff, int) for coeff in poly) or not isinstance(C, int):
        return None
    
    result = [0]
    for i, coeff in enumerate(poly, 1):
        new_coeff = coeff // i
        if new_coeff != 0:
            result.append(new_coeff)
    
    result[0] = C
    
    return result

# Testing the function
if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_integral(poly))
