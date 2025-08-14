import pandas as pd

# Read CSV file
df = pd.read_csv('c:/Users/ECON/Onedrive/Documents/GitHub/FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer/DataSetForPractice/RealEstate-USA.csv', delimiter=",")
print(df)

# Show basic DataFrame info
print(df.info())       # column info, count, type
print(df.dtypes)       # data types of each column
print(df.describe())   # summary stats of numeric columns
print(df.shape)        # total rows and columns

# Print entire DataFrame
print(df.to_string(buf=None))                   
print(df.to_string(columns=['price']))
print(df.to_string(col_space=15))           
print(df.to_string(header=5)) 
print(df.to_string(na_rep="---"))              

# First and last rows
print(df.head(7))       # first 7 rows
print(df.tail(9))       # last 9 rows

# Select single column
print(df["city"])
print(df["street"])

# Select single row by label
row = df.loc[5]
print(row)

# Select multiple rows by label
row2 = df.loc[[3, 5, 7]]
print(row2)

# Select range of rows
slice__Df = df.loc[3:9]
print(slice__Df)

# Filter rows by condition
condition_er = df.loc[df["price"] > 100000]
print(condition_er)

# Filter where city is Adjuntas
adjuntas_row = df.loc[df["city"] == "Adjuntas"]
print(adjuntas_row)

# Multiple conditions
adjuntus_ = df.loc[(df['city'] == 'Adjuntas') & (df['price'] < 180500)]
print(adjuntus_)

# Select specific columns for one row
row3 = df.loc[7, ['city', 'price', 'street', 'zip_code', 'acre_lot']]
print(row3)

# Select columns from 'city' to 'zip_code'
row8 = df.loc[7, "city":"zip_code"]
print(row8)

# Filter + column range
row9 = df.loc[df["city"] == "Adjuntas", "city":"zip_code"]
print(row9)

# Select single row by position
row10 = df.iloc[4]
print(row10)

# Select multiple rows by position
row11 = df.iloc[[6, 8, 14]]
print(row11)

# Select range of rows by position
row12 = df.iloc[5:13]
print(row12)

# Select single column by position
row13 = df.iloc[:, 3]
print(row13)

# Select multiple columns by position
row14 = df.iloc[:, [1, 3, 6]]
print(row14)

# Select range of columns by position
row15 = df.iloc[:, 1:5]
print(row15)

# Select multiple rows + column range
row16 = df.iloc[[6, 8, 14], 3:6]
print(row16)

# Slice rows + column range
row17 = df.iloc[1:6, 2:4]
print(row17)

# Add new row
new_row = {
    'brokered_by': 'Zameen Realty',
    'status': 'Active',
    'price': 750000,
    'bed': 3,
    'bath': 2,
    'acre_lot': 0.5,
    'street': '123 Main St',
    'city': 'New York',
    'state': 'NY',
    'zip_code': '10001',
    'house_size': 1800,
    'prev_sold_date': '07-15-2023'
}
df.loc[len(df.index)] = new_row
print(df.tail())

# Delete single row
df.drop(index=2, inplace=True)
print(df)

# Delete multiple rows
df.drop([4, 5, 6, 7], axis=0, inplace=True)
print(df)

# Delete single column
# df.drop(["house_size"], axis=1, inplace=True)
print(df)

# Delete multiple columns
df.drop(["house_size", "state"], axis=1, inplace=True)
print(df)

# Rename column
df.rename(columns={"state": "state_"}, inplace=True)
print(df)

# Rename row index
df.rename(mapper={3: 5}, axis=0, inplace=True)
print(df)

# Query filter
querry_w = df.query("price < 127400 and city != 'Adjuntas'")
print(querry_w)

# Sort DataFrame by price
sorted_df = df.sort_values(by='price')
print(sorted_df.to_string(index=False))

# Group by city and sum price
grouped_df = df.groupby('city')['price'].sum().reset_index()
print(grouped_df)

# Remove rows with NaN
df_cleaned = df.dropna()
print(df_cleaned)

# Replace NaN with 0
df.fillna(0, inplace=True)
print(df)
