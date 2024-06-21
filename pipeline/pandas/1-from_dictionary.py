#!/usr/bin/env python3

import pandas as pd

def create_dataframe():
    """
    Creates a pandas DataFrame with specified columns and row labels.
    
    Returns:
        pd.DataFrame: The created DataFrame with specified data and labels.
    """
    # Dictionary with data for the DataFrame
    data = {
        'First': [0.0, 0.5, 1.0, 1.5],
        'Second': ['one', 'two', 'three', 'four']
    }
    
    # Row labels
    row_labels = ['A', 'B', 'C', 'D']
    
    # Creating the DataFrame
    df = pd.DataFrame(data, index=row_labels)
    
    return df

# Creating the DataFrame and saving it to the variable df
df = create_dataframe()

# Printing the DataFrame to verify the result
print(df)
