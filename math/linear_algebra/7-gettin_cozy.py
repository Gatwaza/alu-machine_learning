#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1 (list): The first 2D matrix.
        mat2 (list): The second 2D matrix.
        axis (int): The axis along which to concatenate (0 for rows, 1 for columns). Default is 0.

    Returns:
        list: A new 2D matrix containing the concatenated matrices along the specified axis.
              If the two matrices cannot be concatenated, returns None.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        result = mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None

    return result


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6]]
    mat3 = [[7], [8]]
    mat4 = cat_matrices2D(mat1, mat2)
    mat5 = cat_matrices2D(mat1, mat3, axis=1)
    print(mat4)
    print(mat5)
    mat1[0] = [9, 10]
    mat1[1].append(5)
    print(mat1)
    print(mat4)
    print(mat5)
