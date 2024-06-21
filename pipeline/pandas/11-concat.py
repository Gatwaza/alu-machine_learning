#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the specified files
df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Convert 'Timestamp' columns to datetime and set as index
df1['Timestamp'] = pd.to_datetime(df1['Timestamp'], unit='s')
df2['Timestamp'] = pd.to_datetime(df2['Timestamp'], unit='s')

df1.set_index('Timestamp', inplace=True)
df2.set_index('Timestamp', inplace=True)

# Filter bitstamp data up to and including timestamp 1417411920
# Convert the specific timestamp to datetime for comparison
filter_time = pd.to_datetime(1417411920, unit='s')
df2_filtered = df2.loc[:filter_time]

# Concatenate the filtered bitstamp data and coinbase data with keys
df = pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])

# Print the concatenated DataFrame to verify the result
print(df)
