import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = pd.read_csv("C:\\Users\\ECON\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\RealEstate-USA.csv")
print(data)

#properties of DataFrame
print(data.info())
print(data.describe())
print(data.dtypes)
print(data.shape)

#plot setting
plt.figure(figsize=(12,6))
sns.set_theme(style="whitegrid")

# Create a plot
sns.lineplot(x='city', y='price', data=data, errorbar=None)

#graph labeling
plt.title("Average Real Estate price by city")
plt.xlabel("city")
plt.ylabel("price")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


