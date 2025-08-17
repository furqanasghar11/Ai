import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Step 11: Load CSV into DataFrame
file_path = "50_Startups.csv"   # Change path if needed
df = pd.read_csv(file_path)
print("DataFrame Loaded:")
print(df.head())

# Step 12: Explore DataFrame
print("\n--- Data Info ---")
print(df.info())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Data Description ---")
print(df.describe())

print("\n--- Data Shape ---")
print(df.shape)

# Step 13: Independent (X) and Dependent (y) Variables
X = df[["R&D Spend", "Administration", "Marketing Spend"]]  # independent
y = df["Profit"]                                           # dependent

# Step 14: Regression Plots
for col in X.columns:
    sns.regplot(x=df[col], y=y)
    plt.title(f"Regression Plot: {col} vs Profit")
    plt.show()

# Step 15: Correlation Heatmap
corr = df[["R&D Spend", "Administration", "Marketing Spend", "Profit"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Step 16: Train/Test Split (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Step 17: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 18: Intercept
print("\nModel Intercept:", model.intercept_)

# Step 19: Slopes (Coefficients)
print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Step 20: Predictions
y_pred = model.predict(X_test)
print("\nPredicted Profits:", y_pred)

# Step 21: Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- Model Performance ---")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Extra: Plot Actual vs Predicted Profits
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.show()