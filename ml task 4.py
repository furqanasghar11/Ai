import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#load 50_startups csv data
df = pd.read_csv("C:\\Users\\ECON\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\50_Startups (1).csv")
print(df)

#properties of data frame
print(df.info())
print(df.dtypes)
print(df.describe)
print(df.shape)

#Select Independent and Dependent Variables

x = df[["R&D Spend","Administration","Marketing Spend"]]
y = df["Profit"]

 # R&D Spend vs Profit
sns.regplot(x='R&D Spend', y='Profit', data=df)
plt.title("R&D Spend vs Profit")
plt.show()

# Administration vs Profit
sns.regplot(x="Administration",y="Profit",data=df)
plt.title("Administration vs Profit")
plt.show()

#Marketing spend vs Profit
sns.regplot(x="Marketing Spend", y="Profit",data=df)
plt.title("Marketing Spend vs Profit")
plt.show()

#correlation matrix
corr_matrix = df.select_dtypes(include='number').corr()

#plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split
#Split data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=100)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Train our linear regression model with fit() mehtod
regressor.fit(x_train,y_train)

#find and print intercept intercept
print("regressor.intercept_....\n",regressor.intercept_)

#find and print slope
print("regressor.coef_",regressor.coef_)

#Now we pridict Profit based on 3 independent features
y_pred= regressor.predict(x_test)

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





 

