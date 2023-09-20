#!/usr/bin/env python3

"""
This module defines a function to perform matrix multiplication.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication.

    Args:
        mat1 (numpy.ndarray): First input matrix.
        mat2 (numpy.ndarray): Second input matrix.

    Returns:
        numpy.ndarray: Resultant matrix.

    Example:
        >>> mat1 = np.array([[1, 2], [3, 4]])
        >>> mat2 = np.array([[5, 6], [7, 8]])
        >>> np_matmul(mat1, mat2)
        array([[19, 22],
               [43, 50]])
    """
    return np.matmul(mat1, mat2)
