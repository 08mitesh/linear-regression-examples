import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error 

## read csv
df=pd.read_csv('student_scores.csv')

## converted 1D array to 2D array for x since features has to be more than 1
x=df["Hours"].values.reshape(-1, 1)
y=df["Scores"]

## From the entire dataset we are using first 20 records to train model and rest five is used for the testing
x_train, y_train = x[0:20], y[0:20]
x_test, y_test = x[20:], y[20:]

## Providing data to linear regression model
model = LinearRegression().fit(x_train,y_train)

## predicting Y value 
y_predictions= model.predict(x_test)

## plotting values in the visual form - prediction data
plt.plot(x_test,y_predictions,'o')
## plotting values in the visual form - testing data
plt.plot(x_test,y_test,'ro')
