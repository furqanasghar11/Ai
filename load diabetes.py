from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
#load data
diabetes = load_diabetes()

# Convert to DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target
print(df)

# Features (X) and target (y)

x = df.drop(columns=["target"])  # saare features except target
y = df["target"]                  # sirf target column

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42) #70% train and 30% test
print(x_train)
print(y_train)

from sklearn.linear_model import LinearRegression
diabetes_model=LinearRegression()
#Train model
diabetes_model.fit(x_train,y_train)
print(diabetes_model.intercept_)
print(diabetes_model.coef_)

#pridict the model
y_pred=diabetes_model.predict(x_test)
diabetes_df = pd.DataFrame({"Actual": y_test.squeeze(),"Predicted": y_pred.squeeze()})
print(diabetes_df)

#Apply metrics
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
R2=r2_score(y_test,y_pred)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {R2:.2f}')

