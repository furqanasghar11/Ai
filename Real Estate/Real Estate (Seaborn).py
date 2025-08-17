import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:\\Users\\ECON\\OneDrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\RealEstate-USA.csv")
print(df)

# Line plot - city vs price
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
sns.lineplot(data=df, x="city", y="price", estimator="mean", ci=None)
plt.xticks(rotation=45, ha="right")
plt.show()

# Catplot - city vs price
sns.set_theme(style="darkgrid")
sns.catplot(data=df, x="city", y="price", kind="strip", height=6, aspect=2)
plt.xticks(rotation=45, ha="right")
plt.show()

# KDE plot - zip_code vs price
sns.set_theme(style="dark")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="zip_code", y="price", fill=True)
plt.show()

# Scatter plot - zip_code vs price
sns.set_theme(style="ticks")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="zip_code", y="price")
plt.show()

# Bar plot - zip_code vs price
plt.figure(figsize=(14, 6))
sns.barplot(data=df, x="zip_code", y="price")
plt.subplots_adjust(bottom=0.25)
plt.show()

# Heatmap - price ranges vs zip_code
heatmap_data = df[["zip_code", "price"]].dropna()
heatmap_data["price_bin"] = pd.cut(heatmap_data["price"], bins=20)
pivot_table = heatmap_data.pivot_table(index="price_bin", columns="zip_code", aggfunc="size", fill_value=0)

plt.figure(figsize=(14, 8))
sns.heatmap(pivot_table, cmap="YlGnBu")
plt.show()
