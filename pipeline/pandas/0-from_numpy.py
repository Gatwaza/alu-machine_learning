#!/usr/bin/env python3

import numpy as np
import pandas as pd

def from_numpy(array):
    """
    Converts a numpy ndarray to a pandas DataFrame with alphabetically ordered column labels.
    
    Parameters:
        array (np.ndarray): The numpy array to convert to a DataFrame.
    
    Returns:
        pd.DataFrame: The created DataFrame with alphabetically ordered and capitalized columns.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    
    # Determine the number of columns in the array
    num_columns = array.shape[1]
    
    # Generate column labels from 'A' to 'Z'
    column_labels = [chr(i) for i in range(65, 65 + num_columns)]
    
    # Create DataFrame with the generated column labels
    df = pd.DataFrame(array, columns=column_labels)
    
    return df

# Example usage
if __name__ == "__main__":
    # Creating a sample numpy array
    sample_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Converting the numpy array to a pandas DataFrame
    df = from_numpy(sample_array)
    print(df)
