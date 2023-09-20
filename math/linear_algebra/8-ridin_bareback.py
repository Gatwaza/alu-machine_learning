#!/usr/bin/env python3
"""
This module defines a function to multiply a matrix.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication.

    Args:
        mat1 (list): The first 2D matrix.
        mat2 (list): The second 2D matrix.

    Returns:
        list: A new 2D matrix containing
        the result of the matrix multiplication.
        If the two matrices cannot be multiplied, returns None.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = [[sum(a*b for a, b in zip(row, col))
               for col in zip(*mat2)] for row in mat1]
    return result


if __name__ == "__main__":
    mat1 = [[1, 2],
            [3, 4],
            [5, 6]]
    mat2 = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    print(mat_mul(mat1, mat2))
