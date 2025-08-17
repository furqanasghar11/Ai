import numpy as np
import pandas as pd

#  Load the dataset from the specified path
df = pd.read_csv("C:\\Users\\ECON\\OneDrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\startup_growth_investment_data.csv")

#  Inspect the first few rows to understand the structure
print(" Head of the dataset:")
print(df.head())

#  Display the column names
print("\n Column Names:")
print(df.columns)

#  Summary information about the dataset (data types, nulls, memory)
print("\n Dataset Info:")
df.info()

#  Descriptive statistics for numeric columns
print("\n Descriptive Statistics:")
print(df.describe())

#  Dataset dimensions: (rows, columns)
print("\n Dataset Shape:")
print(df.shape)

#  Display specific subsets of the dataset
print("\n First 7 rows:")
print(df.head(7))

print("\n Last 9 rows:")
print(df.tail(9))

#  Display selected columns
print("\n 'Country' column:")
print(df['Country'])

print("\n 'Year Founded' column:")
print(df['Year Founded'])

print("\n Selected columns ['Country', 'Year Founded', 'Startup Name']:")
print(df[['Country', 'Year Founded', 'Startup Name']])

#  Access specific rows using .loc
print("\n Row at index 5:")
print(df.loc[5])

print("\n Rows at index 3, 5, and 7:")
print(df.loc[[3, 5, 7]])

print("\n Rows from index 3 to 9:")
print(df.loc[3:9])

#  Filter rows where investment amount is greater than 100,000
print("\n Startups with Investment Amount > 100,000:")
print(df[df['Investment Amount (USD)'] > 100000])

#  Filter based on valuation
print("\n Startups with Valuation <= 180,500:")
print(df[df['Valuation (USD)'] <= 180500])

#  Select a specific row and columns
print("\n Row 7 - selected columns:")
print(df.loc[7, ['Country', 'Year Founded', 'Startup Name', 'Investment Amount (USD)']])

#  Slice columns from 'Country' to 'Year Founded'
print("\n Slice columns from 'Country' to 'Year Founded':")
print(df.loc[:, 'Country':'Year Founded'])

#  Using .iloc for positional indexing
print("\n Row at position 5:")
print(df.iloc[5])

print("\n Rows at positions 7, 9, and 15:")
print(df.iloc[[7, 9, 15]])

print("\n Rows 5 to 12:")
print(df.iloc[5:13])

print("\n Column at index 3:")
print(df.iloc[:, 3])

print("\n Columns at index 2, 4, 8:")
print(df.iloc[:, [2, 4, 8]])

print("\n Slice columns 2 to 4:")
print(df.iloc[:, 2:5])

#  Drop specific rows by index
print("\n Drop row 2:")
df_new = df.drop(2)
print(df_new)

print("\n Drop rows 4, 5, 6, 7:")
df_new2 = df.drop([4, 5, 6, 7])
print(df_new2)

#  Drop a column
print("\n Drop 'Year Founded' column:")
print(df.drop('Year Founded', axis=1))

#  Rename column
print("\n Rename 'Country' to 'country name':")
print(df.rename(columns={'Country': 'country name'}))

#  Rename row index
print("\n Rename row index 3 to 5:")
print(df.rename(index={3: 5}))

#  Sort the dataset by country name in descending order
print("\n Sort by 'Country' descending:")
print(df.sort_values('Country', ascending=False))

#  Group by 'Country' and sum 'Year Founded' (not meaningful but shows grouping)
print("\n Group by 'Country' and sum 'Year Founded':")
print(df.groupby('Country')['Year Founded'].sum())

#  Drop rows with missing values
print("\n Drop rows with missing values:")
print(df.dropna())

# Fill missing values with 0
print("\n Fill missing values with 0:")
df_filled = df.fillna(0)
print(df_filled)