#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the specified file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Step 1: Remove rows where the 'Close' column is NaN
df = df.dropna(subset=['Close'])

# Step 2: Print the first few rows of the cleaned DataFrame to verify the changes
print(df.head())
