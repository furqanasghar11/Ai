import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:\\Users\\ECON\\OneDrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\DataSetForPractice\\startup_growth_investment_data.csv", delimiter=",")
print(df)

# First hunderd rows make a graph 
dffilter = df.head(100)
print("First hundered rows:",dffilter)

# Customize the theme
sns.set_theme(style="darkgrid", rc={"axes.facecolor":"grey", "grid.color": "white"})

# Create a line plot Graph
# Graph 1
g = sns.displot(data= dffilter, x = "Investment Amount (USD)", y = "Valuation (USD)")
g.figure.suptitle("Displot graph between Investment Amount (USD) and Valuation (USD)")
plt.show()


#kind='kde'
# Graph 2
g = sns.displot(data= dffilter, x = "Investment Amount (USD)", y = "Valuation (USD)")
g.figure.suptitle("Displot Graph between Investment Amount (USD) and Valuation (USD):")
plt.show()

# Graph 3
g = sns.kdeplot(data=dffilter, x = "Investment Amount (USD)", y="Valuation (USD)")
g.figure.suptitle("Kdeplot Graph between Investment Amount (USD) and Valuation (USD):")
plt.show()

# Graph 4
g = sns.histplot(data=dffilter, x = "Investment Amount (USD)", y="Valuation (USD)")
g.figure.suptitle("Histplot Graph between Investment Amount (USD) and Valuation (USD):")
plt.show()

# Graph 5
g = sns.scatterplot(data=dffilter, x = "Investment Amount (USD)", y="Valuation (USD)")
g.figure.suptitle("Scatterplot Graph between Investment Amount (USD) and Valuation (USD):")
plt.show()

# Graph 6
g = sns.barplot(data=dffilter, x = "Investment Amount (USD)", y="Valuation (USD)")
g.figure.suptitle("Barplot Graph between Investment Amount (USD) and Valuation (USD):")
plt.show()

# Graph 7
g = sns.catplot(data=dffilter, x = "Valuation (USD)", y="Investment Amount (USD)")
g.figure.suptitle("Catplot Graph between Investment Amount (USD) and Valuation (USD):")
plt.show()

# Graph 8
corr = dffilter[["Investment Amount (USD)","Valuation (USD)"]].corr()
g = sns.heatmap(corr,annot=True,cmap="coolwarm")
g.figure.suptitle("Heatmap Graph between Investment Amount (USD) and Valuation (USD):")
plt.show()
