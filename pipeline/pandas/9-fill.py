#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the specified file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Step 1: Remove the 'Weighted_Price' column
df = df.drop(columns=['Weighted_Price'])

# Step 2: Fill missing values in 'Close' with the previous row's value (forward fill)
df['Close'].fillna(method='ffill', inplace=True)

# Step 3: Fill missing values in 'High', 'Low', and 'Open' with the current row's 'Close' value
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)

# Step 4: Fill missing values in 'Volume_(BTC)' and 'Volume_(Currency)' with 0
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# Print the first few rows and the last few rows of the updated DataFrame to verify the changes
print(df.head())
print(df.tail())
