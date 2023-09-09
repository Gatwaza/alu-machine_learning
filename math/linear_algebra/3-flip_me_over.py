#!/usr/bin/env python3

def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
    matrix (list): The input matrix.

    Returns:
    list: A new matrix that is the transpose of the input matrix.
    """
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    # Create a new matrix with swapped rows and columns
    transpose_matrix = [[matrix[j][i] for j in range(num_rows)] for i in range(num_cols)]
    
    return transpose_matrix
