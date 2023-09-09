#!/usr/bin/env python3

def cat_arrays(arr1, arr2):
    """
    Concatenates two arrays.

    Args:
        arr1 (list): The first array.
        arr2 (list): The second array.

    Returns:
        list: A new list containing the elements of arr1 followed by the elements of arr2.
    """
    result = arr1 + arr2
    return result

if __name__ == "__main__":
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [6, 7, 8]
    print(cat_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
