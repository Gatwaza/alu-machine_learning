#!/usr/bin/env python3
"""
Function that adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise.

    Args:
        mat1 (list): The first 2D matrix.
        mat2 (list): The second 2D matrix.

    Returns:
        list: A new 2D matrix containing the element-wise sum of mat1 and mat2.
              If mat1 and mat2 are not the same shape, returns None.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    result = [[a + b for a, b in zip(row1, row2)]
              for row1, row2 in zip(mat1, mat2)]
    return result


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(add_matrices2D(mat1, mat2))
    print(mat1)
    print(mat2)
    print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
