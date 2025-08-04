import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Let's read the CSV file and package it into a DataFrame:
df = pd.read_csv("C:\\Users\\ECON\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\number-of-registered-medical-and-dental-doctors-by-gender-in-pakistan (1).csv",delimiter=",",index_col="Years")
print(df)

#Properties of pandas
print(df.info())
print(df.dtypes)
print(df.describe())
print(df.shape)

# Clean comma-containing columns
cols_to_clean = ["Female Doctors", "Male Doctors", "Total Doctors", "Female Dentists", "Total Dentists"]
for col in cols_to_clean:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

#Define input and ouput
x = df["Female Doctors"].values.reshape(-1,1)
y = df["Female Dentists"].values.reshape(-1,1)

print("x is :",x)
print("x is :",y)

from sklearn.model_selection import train_test_split
#split data
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=100)
print(x_train)
print(y_train)

from sklearn.linear_model import LinearRegression
#train model
regressor = LinearRegression()

regressor.fit(x_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)

def calc(slope,intercept,femaledoctors):
    return slope*femaledoctors+intercept

value = calc(regressor.coef_,regressor.intercept_,3146)
print(value)
#pridict 
y_pred = regressor.predict(x_test)
df_doctor = pd.DataFrame({"Actual": y_test.squeeze(),"Predicted": y_pred.squeeze()})
print(df_doctor)
# Plot results
plt.scatter(x_test, y_test, color='blue', label="Actual")
plt.plot(x_test, y_pred, color='red', label="Predicted", linewidth=2)
plt.xlabel("Female Doctors")
plt.ylabel("Female Dentists")
plt.title("Linear Regression: Female Doctors vs Dentists")
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')
