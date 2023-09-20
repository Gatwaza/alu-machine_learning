#!/usr/bin/env python3
"""
This module defines a function to perform element-wise
 addition, subtraction, multiplication, and division on numpy.ndarrays.
"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise
         addition, subtraction, multiplication, and division.

    Args:
        mat1 (numpy.ndarray): First input array.
        mat2 (numpy.ndarray): Second input array.

    Returns:
        tuple of numpy.ndarray:
         A tuple containing the element-wise sum, difference,
        product, and quotient, respectively.

    Example:
        >>> mat1 = np.array([[1, 2], [3, 4]])
        >>> mat2 = np.array([[5, 6], [7, 8]])
        >>> np_elementwise(mat1, mat2)
        (array([[ 6,  8],
               [10, 12]]), array([[-4, -4],
               [-4, -4]]), array([[ 5, 12],
               [21, 32]]), array([[0.2       , 0.33333333],
               [0.42857143, 0.5       ]]))
    """
    sum_result = mat1 + mat2
    diff_result = mat1 - mat2
    prod_result = mat1 * mat2
    quot_result = mat1 / mat2
    return sum_result, diff_result, prod_result, quot_result
