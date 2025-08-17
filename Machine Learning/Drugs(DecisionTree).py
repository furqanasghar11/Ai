import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier   # Decision Tree for classification
from sklearn import metrics                       # For model accuracy check
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("C:\\Users\\ECON\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\drug200.csv")

# Dataset basic info
print(df)
print(df.info())
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.head())
print(df.tail())

# Feature and Target
X = df[["Age","Sex","BP","Cholesterol","Na_to_K"]]   # Input features
y = df[["Drug"]]                                     # Target column

# Data Preprocessing
# Convert categorical values into numeric
X.loc[:, "Sex"] = X["Sex"].map({"F": 0, "M": 1})
X.loc[:, "BP"] = X["BP"].map({"High": 0, "Low": 1})
X.loc[:, "Cholesterol"] = X["Cholesterol"].map({"High": 0, "Normal": 1})

# Convert columns into float
X.loc[:, "Age"] = X["Age"].astype(float)
X.loc[:, "NaToK"] = X["Na_to_K"].astype(float)

# Train Test Split
# 70 percent training data, 30 percent testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Model Training
model = DecisionTreeClassifier(max_depth=3)  # Decision Tree with max depth = 3
model.fit(X_train, y_train)                  # Train model

# Prediction
y_pred = model.predict(X_test)               # Predict test set

# Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Plot Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, filled=True)
plt.show()
