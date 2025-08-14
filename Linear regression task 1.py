# 📦 Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ Load the CSV into DataFrame
df = pd.read_csv("C:\\Users\\ECON\\\Onedrive\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\Real_Estate_Sales_2001-2022_GL-Short.csv",delimiter=",",index_col="Serial Number",na_filter=False)
print(df)

#DataFrame basic properties
print(df.info())        
print(df.dtypes)        
print(df.describe())    
print(df.shape)

#Prepare feature (X) and target (y)
x = df["Assessed Value"].values.reshape(-1, 1)
y = df["Sale Amount"].values.reshape(-1, 1) 

print("x is :", x)
print("y is :", y)

# 4️⃣ Split data (90% training, 10% testing)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=100
)
print("Test X values:\n", x_test)
print("Test Y values:\n", y_test)

# 5️⃣ Train Linear Regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# 6️⃣ Print model parameters
print("Intercept:", regressor.intercept_)
print("Slope:", regressor.coef_)

# #calculate assessed_value based on slope & intercept
def calc(slope, intercept, assessed_value):
    return slope * assessed_value + intercept

# Example calculation
value_manual = calc(regressor.coef_, regressor.intercept_, 217640.00)
print("Manual calc prediction:", value_manual)

# Predict directly using the model
value_model = regressor.predict([[217640.00]])
print("Model prediction:", value_model)

# Predict for test data
y_pred = regressor.predict(x_test)

# Create comparison DataFrame
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

#Model evaluation metrics
mae = mean_absolute_error(y_test, y_pred)    # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)     # Mean Squared Error
rmse = np.sqrt(mse)                          # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)                # R² Score

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

#Graph: Regression line + data points
plt.figure(figsize=(8,6))

# Scatter plot for training data (blue)
plt.scatter(x_train, y_train, color="blue", label="Training Data")

# Scatter plot for testing data (red)
plt.scatter(x_test, y_test, color="red", label="Testing Data")

# Regression line (green)
plt.plot(x, regressor.predict(x), color="green", linewidth=2, label="Regression Line")

# Labels and title
plt.xlabel("Assessed Value")
plt.ylabel("Sale Amount")
plt.title("Linear Regression: Assessed Value vs Sale Amount")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
