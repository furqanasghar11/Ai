import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Let's read the CSV file and package it into a DataFrame:
df = pd.read_csv("C:\\Users\\ECON\\OneDrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\number-of-registered-medical-and-dental-doctors-by-gender-in-pakistan.csv", delimiter=",", index_col="Years")
print(df)

#Properties of pandas
print(df.info())
print(df.dtypes)
print(df.describe())
print(df.shape)

#Clean numeric columns (remove commas)
for col in ["Female Doctors", "Female Dentists"]:
    df[col] = df[col].replace({",": ""}, regex=True).astype(float)

# Convert columns to numpy arrays for sklearn
x = df[["Female Doctors"]].values
y = df[["Female Dentists"]].values
print("x is :", x)
print("y is :", y)

from sklearn.model_selection import train_test_split
#split data 70% train and 30% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LinearRegression
# Train linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

#Print intercept and slope
print("Intercept:", model.intercept_)
print("Slope:", model.coef_)

# Calculate price based on slope & intercept
def calc(slope, intercept, TotalDoctors):
    return slope *TotalDoctors + intercept

# Test custom function
value = calc(model.coef_, model.intercept_, 7467)
print(value)

# Predict directly using model
value = model.predict([[7467]])
print(value)

#Predict test data
y_pred = model.predict(x_test)

# Create DataFrame to compare actual vs predicted
data = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(data)

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

#Plot graph
plt.figure(figsize=(8,6))
plt.scatter(x_test, y_test, color='blue', label="Actual Data")
plt.plot(x_test, y_pred, color='red', linewidth=2, label="Regression Line")
plt.title("Female Doctors vs Female Dentists (Prediction)")
plt.xlabel("Female Doctors")
plt.ylabel("Female Dentists")
plt.legend()
plt.grid(True)
plt.show()