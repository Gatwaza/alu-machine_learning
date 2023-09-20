#!/usr/bin/env python3
def np_transpose(matrix):
    """
    Transpose the input matrix.

    Args:
        matrix (list of lists): Input array.

    Returns:
        list of lists: Transposed array.

    Example:
        >>> matrix = [[1, 2, 3], [4, 5, 6]]
        >>> np_transpose(matrix)
        [[1, 4], [2, 5], [3, 6]]
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
