import pandas as pd

df = pd.read_csv('c:/Users/ECON/Documents/GitHub/FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer/DataSetForPractice/RealEstate-USA.csv',delimiter=",")
print(df)

#properties of DataFrame

print(df.info())
print(df.dtypes)
print(df.describe())
print(df.shape)

# DataFrame method .to_string()
print(df.to_string(buf=None))
print(df.to_string(columns=['price']))
print(df.to_string(col_space=15))
print(df.to_string(header=5))
print(df.to_string(na_rep="---"))


#first 7 header print
print(df.head(7))

#tail bottom 9 row print
print(df.tail(9))

print(df["city"])
print(df["street"])

#Selecting a single row using .loc
row = df.loc[5]
print(row)

#Selecting multiple rows using .loc
row2 = df.loc[[3,5,7]]
print(row2)

#Selecting a slice of rows using .loc
slice__Df=df.loc[3:9]
print(slice__Df)

# question no 11
condition_er = df.loc[df["price"]>100000]
print(condition_er)

#question no 12
adjuntas_row=df.loc[df["city"]== "Adjuntas"]
print(adjuntas_row)

#question no 13
adjuntus_= df.loc[(df['city'] == 'Adjuntas') & (df['price'] < 180500)]
print(adjuntus_)


#question no 14
row3 = df.loc[7, ['city', 'price', 'street', 'zip_code', 'acre_lot']]
print(row3)

# question no 15
row8=df.loc[7,"city":"zip_code"]
print(row8)

#question no 16
row9=df.loc[df["city"]=="Adjuntas","city":"zip_code"]
print(row9)

# question no 17
row10 = df.iloc[4]
print(row10)

#question no 18
row11=df.iloc[[6,8,14]]
print(row11)

#question no 19
row12 = df.iloc[5:13]
print(row12)

#Question no 20
row13 =df.iloc[:, 3]
print(row13)

#question no 21
row14 =df.iloc[:,[1,3,6] ]
print(row14)

#question no 22
row15 =df.iloc[:, 1:5]
print(row15)

#QUESTION No 23
row16=df.iloc[[6,8,14],3:6]
print(row16)

#question no 24
row17=(df.iloc[1:6,2:4])
print(row17)

#QUESTION NO 25
# add new row to dataframe
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

#question no 26
df.drop(index=2, inplace=True)
print(df)

#quetion no 27
df.drop([4,5,6,7],axis=0,inplace=True)
print(df)

#question no 28
# df.drop(["house_size"],axis = 1,inplace=True)
print(df)

#QUESTION No 29
df.drop(["house_size","state"],axis=1,inplace=True)
print(df)

#question no 30
df.rename(columns={"state":"state_"},inplace=True)
print(df)

#question no 31
df.rename(mapper={3:5},axis=0,inplace=True)
print(df)

#question no 32

querry_w=df.query("price < 127400 and city != 'Adjuntas'")
print(querry_w)

#QUEStion no 33
# sort DataFrame by price in ascending order
sorted_df = df.sort_values(by='price')
print(sorted_df.to_string(index=False))

#question no 34
# group the DataFrame by the location_id column and
# calculate the sum of price for each category
grouped_df = df.groupby('city')['price'].sum().reset_index()
print(grouped_df)

#question no 35
# use dropna() to remove rows with any missing values
df_cleaned=df.dropna()
print(df_cleaned)

#question no 36
# filling NaN values with 0
df.fillna(0, inplace=True)
print(df)