import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

df_FastFoodResturant = pd.read_csv("C:\\Users\\ECON\\OneDrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\FastFoodRestaurants.csv", delimiter=",")

print(df_FastFoodResturant)

 
 
#kind='hist'
# Graph 1
g = sns.displot(data= df_FastFoodResturant, x = "longitude", y = "latitude")
g.figure.suptitle("Displot graph between longitude and latitude")
plt.show()

#kind='kde'
# Graph 2
g = sns.displot(data= df_FastFoodResturant, x = "longitude", y = "latitude")
g.figure.suptitle("Displot Graph between longitude and bath:")
plt.show()

# Graph 3
g = sns.kdeplot(data=df_FastFoodResturant, x = "longitude", y="bath")
g.figure.suptitle("Kdeplot Graph between longitude and latitude:")
plt.show()

# Graph 4
g = sns.histplot(data=df_FastFoodResturant, x = "longitude", y="latitude")
g.figure.suptitle("Histplot Graph between longitude and latitude:")
plt.show()

# Graph 5
g = sns.scatterplot(data=df_FastFoodResturant, x = "longitude", y="latitude")
g.figure.suptitle("Scatterplot Graph between longitude and latitude:")
plt.show()

# Graph 6
g = sns.barplot(data=df_FastFoodResturant, x = "longitude", y="latitude")
g.figure.suptitle("Barplot Graph between longitude and latitude:")
plt.show()

# Graph 7
g = sns.catplot(data=df_FastFoodResturant, x = "latitude", y="longitude")
g.figure.suptitle("Catplot Graph between longitude and latitude:")
plt.show()

# Graph 8
corr = df_FastFoodResturant[["longitude","latitude"]].corr()
g = sns.heatmap(corr,annot=True,cmap="coolwarm")
g.figure.suptitle("Heatmap Graph between longitude and latitude:")
plt.show()