#!/usr/bin/env python3
"""
Function that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): The first array.
        arr2 (list): The second array.

    Returns:
        list: A new list containing the element-wise sum of arr1 and arr2.
              If arr1 and arr2 are not the same shape, returns None.
    """
    if len(arr1) != len(arr2):
        return None

    result = [a + b for a, b in zip(arr1, arr2)]
    return result


if __name__ == "__main__":
    arr1 = [1, 2, 3, 4]
    arr2 = [5, 6, 7, 8]
    print(add_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
    print(add_arrays(arr1, [1, 2, 3]))
