import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read CSV file into a DataFrame
df = pd.read_csv(
    "C:\\Users\\ECON\\Onedrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\Week2\\zameencom-property-data-By-Kaggle-Short.csv", delimiter=";",index_col="property_id")
print(df)

#Check DataFrame properties
print(df.info())        # Overview of dataset
print(df.dtypes)        # Data types of each column
print(df.describe())    # Statistical summary
print(df.shape)         # Shape of DataFrame (rows, columns)

# Prepare X (bedrooms) and Y (price) for ML model
x = df["bedrooms"].values.reshape(-1, 1)  
y = df["price"].values.reshape(-1, 1)

print("x is :", x)
print("y is :", y)

#Split data into 75% training and 25% testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)
print(x_train)
print(y_train)

# Create and train Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Print intercept and slope
print("Intercept:", regressor.intercept_)
print("Slope:", regressor.coef_)

#calculate price based on slope & intercept
def calc(slope, intercept, bedrooms):
    return slope * bedrooms + intercept

# Test custom function
value = calc(regressor.coef_, regressor.intercept_, 5)
print("Predicted price for 5 bedrooms (manual function):", value)

# Predict directly using model
value = regressor.predict([[5]])
print("Predicted price for 5 bedrooms (model.predict):", value)

#Predict prices for test data
y_pred = regressor.predict(x_test)

# Create DataFrame to compare actual vs predicted
df_price = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_price)

# Model evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)    # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)     # Mean Squared Error
rmse = np.sqrt(mse)                          # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)                # R² Score

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')

# Plot the regression line with training & testing data
plt.figure(figsize=(8,6))

# Scatter plot for training data (blue)
plt.scatter(x_train, y_train, color="blue", label="Training Data")

# Scatter plot for testing data (red)
plt.scatter(x_test, y_test, color="red", label="Testing Data")

# Regression line (green)
plt.plot(x, regressor.predict(x), color="green", linewidth=2, label="Regression Line")

# Labels, title, legend
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price")
plt.title("Linear Regression: Bedrooms vs Price")
plt.legend()
plt.grid(True)

# Show graph
plt.show()
