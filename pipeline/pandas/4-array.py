#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the specified file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Select the last 10 rows of the columns 'High' and 'Close'
last_10_rows = df[['High', 'Close']].tail(10)

# Convert the selected rows to a numpy ndarray
A = last_10_rows.to_numpy()

# Print the resulting numpy ndarray
print(A)
