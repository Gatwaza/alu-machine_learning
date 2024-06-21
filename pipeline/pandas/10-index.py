#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the specified file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Step 1: Convert 'Timestamp' to datetime format (if it's in UNIX time, it will be integers/strings)
# Assuming 'Timestamp' is in UNIX time and needs conversion
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

# Step 2: Set the 'Timestamp' column as the DataFrame index
df.set_index('Timestamp', inplace=True)

# Print the last few rows of the updated DataFrame to verify the changes
print(df.tail())
