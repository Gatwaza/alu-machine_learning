#!/usr/bin/env python3
"""
This module defines a function to calculate the shape of a numpy.ndarray.
"""

def np_shape(matrix):
    """
    Calculate the shape of a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): Input array.

    Returns:
        tuple of int: A tuple representing the dimensions of the input array.

    Example:
        >>> matrix = np.array([[1, 2, 3], [4, 5, 6]])
        >>> np_shape(matrix)
        (2, 3)
    """
    return matrix.shape
