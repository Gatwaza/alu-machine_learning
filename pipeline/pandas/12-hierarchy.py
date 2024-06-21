#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the specified files
df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Convert 'Timestamp' columns to datetime
df1['Timestamp'] = pd.to_datetime(df1['Timestamp'], unit='s')
df2['Timestamp'] = pd.to_datetime(df2['Timestamp'], unit='s')

# Set the 'Timestamp' column as the index
df1.set_index('Timestamp', inplace=True)
df2.set_index('Timestamp', inplace=True)

# Define the filter range
start_time = pd.to_datetime(1417411980, unit='s')
end_time = pd.to_datetime(1417417980, unit='s')

# Filter both DataFrames to include only the specified timestamp range
df1_filtered = df1.loc[start_time:end_time]
df2_filtered = df2.loc[start_time:end_time]

# Concatenate the filtered DataFrames with keys
df = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp', 'coinbase'])

# Swap the levels of the MultiIndex to make 'Timestamp' the first level
df = df.swaplevel(i=0, j=1)

# Sort the DataFrame by the 'Timestamp' index
df = df.sort_index(level='Timestamp')

# Print the concatenated and reindexed DataFrame to verify the result
print(df)
