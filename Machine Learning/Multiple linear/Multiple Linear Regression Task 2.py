import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Load California Housing data 

df = pd.read_csv("C:\\Users\\ECON\\Onedrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\housing.csv")
print(df)

#properties of data frame
print(df.info())
print(df.dtypes)
print(df.describe)
print(df.shape)

df = df.dropna(subset=["housing_median_age","total_bedrooms","latitude","median_house_value"])

#Select Independent and DEPENDENT variables
x = df[["housing_median_age","total_bedrooms","latitude"]]
y = df["median_house_value"]

#housing_median_age vs median_house_value
sns.regplot(x="housing_median_age",y="median_house_value",data=df)
plt.title("housing_median_age vs median_house_value")
plt.show()

#total_bedrooms vs median_house_value
sns.regplot(x="total_bedrooms",y="median_house_value",data=df)
plt.title("total_bedrooms vs median_house_value ")
plt.show()

#Correlation matrix
corr_matrix = df.select_dtypes(include='number').corr()
#plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

#train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

#Intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

#Predict on test data
y_pred = model.predict(x_test)

#print pridicted vs actual value
result = pd.DataFrame({"Actual Profit":y_test, "Predicted Profit":y_pred})
print(result)

#calculate the MAE and MSE and also the square root of mse
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')


