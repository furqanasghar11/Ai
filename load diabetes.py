from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Load dataset
diabetes = load_diabetes()

#Convert to DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target
print(df)

#Features (X) and target (y)
x = df.drop(columns=["target"])
y = df["target"]

#Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
print(x_train)
print(y_train)

#Train Linear Regression model
diabetes_model = LinearRegression()
diabetes_model.fit(x_train, y_train)

#Print intercept and coefficients
print("Intercept:", diabetes_model.intercept_)
print("Coefficients:", diabetes_model.coef_)

#Predict values
y_pred = diabetes_model.predict(x_test)

#Compare Actual vs Predicted
diabetes_df = pd.DataFrame({"Actual": y_test.squeeze(), "Predicted": y_pred.squeeze()})
print(diabetes_df)

#Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
R2 = r2_score(y_test, y_pred)

print(f"Mean absolute error: {mae:.2f}")
print(f"Mean squared error: {mse:.2f}")
print(f"Root mean squared error: {rmse:.2f}")
print(f"R2 Score: {R2:.2f}")

#Scatter plot of Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linewidth=2)  # Perfect prediction line
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs Predicted - Diabetes Regression")
plt.grid(True)
plt.show()

# Bar chart comparison for first 20 samples
comparison_df = diabetes_df.head(20)  # Take first 20 test samples
comparison_df.plot(kind="bar", figsize=(10,6))
plt.title("Actual vs Predicted Values (First 20 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.grid(True)
plt.show()

