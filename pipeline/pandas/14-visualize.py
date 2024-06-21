#!/usr/bin/env python3

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data from the specified file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove 'Weighted_Price' column
df.drop(columns=['Weighted_Price'], inplace=True)

# Rename 'Timestamp' to 'Date' and convert to datetime
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
df.drop(columns=['Timestamp'], inplace=True)

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Fill missing values
df['Close'].fillna(method='ffill', inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# Filter data from 2017 onwards
df = df['2017':]

# Resample data at daily intervals and apply aggregation functions
df_resampled = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(df_resampled.index, df_resampled['High'], label='High', marker='o', linestyle='-')
plt.plot(df_resampled.index, df_resampled['Low'], label='Low', marker='o', linestyle='-')
plt.plot(df_resampled.index, df_resampled['Open'], label='Open', marker='o', linestyle='-')
plt.plot(df_resampled.index, df_resampled['Close'], label='Close', marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Prices and Volume (2017 onwards)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
