#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the specified file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Step 1: Sort the DataFrame in reverse chronological order based on the 'Timestamp' column
df.sort_values(by='Timestamp', ascending=False, inplace=True)

# Step 2: Transpose the DataFrame (rows become columns and columns become rows)
df = df.transpose()

# Print the last 8 rows of the resulting transposed DataFrame
print(df.tail(8))
