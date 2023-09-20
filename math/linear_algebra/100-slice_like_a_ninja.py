#!/usr/bin/env python3

"""
This module defines a function to slice a matrix along specific axes.
"""


def np_slice(matrix, axes={}):
    """
    Slice a matrix along specific axes.

    Args:
        matrix (numpy.ndarray): Input matrix.
        axes (dict): Dictionary where the key is an axis to slice along and the
            value is a tuple representing the slice to make along that axis.

    Returns:
        numpy.ndarray: Sliced matrix.

    Example:
        >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> axes = {0: (0, 2), 1: (1, 3)}
        >>> np_slice(matrix, axes)
        array([[2, 3],
               [5, 6]])
    """
    return matrix[tuple(slice(start, end)
                        for axis, (start, end) in axes.items())]
