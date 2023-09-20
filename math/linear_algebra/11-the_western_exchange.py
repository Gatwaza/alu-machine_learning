#!/usr/bin/env python3
def np_transpose(matrix):
    """
    Transpose the input matrix.

    Args:
        matrix (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Transposed array.

    Example:
        >>> matrix = np.array([[1, 2, 3], [4, 5, 6]])
        >>> np_transpose(matrix)
        array([[1, 4],
               [2, 5],
               [3, 6]])
    """
    return matrix.T
