#!/usr/bin/env python3

"""
This module defines a function to slice a matrix along specific axes.
"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes.

    Args:
        matrix (numpy.ndarray): Input matrix.
        axes (dict): Dictionary where the key is an axis to slice along and
                     the value is a tuple representing the slice to make along that axis.

    Returns:
        numpy.ndarray: Sliced matrix.
    """
    # Copy the original matrix to avoid modifying it
    result = [row[:] for row in matrix]

    # Apply the specified slices along the specified axes
    for axis, slice_range in axes.items():
        if len(slice_range) == 1:
            slice_range = slice_range[0]
            if slice_range is not None:
                result = result[slice(slice_range)]
        elif len(slice_range) == 2:
            start, stop = slice_range
            if start is not None or stop is not None:
                result = [row[start:stop] for row in result]
        elif len(slice_range) == 3:
            start, stop, step = slice_range
            if start is not None or stop is not None or step is not None:
                result = [row[start:stop:step] for row in result]

    return result

