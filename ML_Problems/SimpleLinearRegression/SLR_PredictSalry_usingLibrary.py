from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plt.rcParams['figure.figsize'] = (20.0, 10.0)
#Reading data
data = pd.read_csv('Salary_Data.csv')
print (data)
#data.head()

#collecting x and y
x = data['YearsExperience'].values
y = data['Salary'].values
plt.scatter(x,y)
#total no of values
m = len(x)
#can not use rank 1 metrics in scikit learn
X = x.reshape((m, 1))
Y = y.reshape((m, 1))

#creating model
reg = LinearRegression()

#fitting trainig data
reg = reg.fit(X,Y)

#Y prediction
y_pred = reg.predict(X)

#Calculating RMSE and R2 score
#mse = mean_squared_error(y,y_pred)
#rmse = np.sqrt(mse)
r2_score = reg.score(X,Y)

#print(np.sqrt(mse))
print(r2_score)
plt.plot(X,y_pred)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()