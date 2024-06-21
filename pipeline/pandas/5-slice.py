#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the specified file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Print the column names to inspect the DataFrame structure
print("Column names in the DataFrame:")
print(df.columns)

# Assuming we find the correct name for 'Volume_BTC', let's proceed with slicing
# Check the correct name for the volume column from the printed list
# For this example, let's assume the correct column is 'Volume_(BTC)'

# Select the columns 'High', 'Low', 'Close', and the actual volume column and take every 60th row
df = df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]

# Print the last few rows of the resulting DataFrame
print(df.tail())
