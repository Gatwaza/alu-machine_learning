#!/usr/bin/env python3

import pandas as pd

def from_file(filename, delimiter):
    """
    Loads data from a file into a pandas DataFrame.

    Parameters:
        filename (str): The path to the file to load.
        delimiter (str): The column separator/delimiter used in the file.

    Returns:
        pd.DataFrame: The DataFrame containing the loaded data.
    """
    try:
        # Load the data into a DataFrame using the specified delimiter
        df = pd.read_csv(filename, delimiter=delimiter)
    except Exception as e:
        raise ValueError(f"Error loading file '{filename}': {e}")
    
    return df

# Example usage
if __name__ == "__main__":
    # Specify the file path and delimiter for testing
    file_path = 'data.csv'  # Replace with your actual file path
    delim = ','  # Replace with your actual delimiter, e.g., ',', '\t', ';'
    
    # Load the data and print the DataFrame
    try:
        df = from_file(file_path, delim)
        print(df)
    except ValueError as e:
        print(e)
