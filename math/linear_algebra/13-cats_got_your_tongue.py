#!/usr/bin/env python3

"""
This module defines a function to concatenate
 two matrices along a specific axis.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specific axis.

    Args:
        mat1 (numpy.ndarray): First input array.
        mat2 (numpy.ndarray): Second input array.
        axis (int, optional): The axis along
        which the matrices will be concatenated.
            Default is 0.

    Returns:
        numpy.ndarray: Concatenated array.

    Example:
        >>> mat1 = np.array([[1, 2], [3, 4]])
        >>> mat2 = np.array([[5, 6], [7, 8]])
        >>> np_cat(mat1, mat2, axis=0)
        array([[1, 2],
               [3, 4],
               [5, 6],
               [7, 8]])
    """
    return np.concatenate((mat1, mat2), axis=axis)
