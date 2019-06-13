import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('bike_sharing.csv')
print(data)
x = data['cnt'].values
y = data['temp'].values
m = len(x)

X = x.reshape((m, 1))
Y = y.reshape((m, 1))
reg = LinearRegression()
reg = reg.fit(X,Y)
y_pred = reg.predict(X)
r2_score = reg.score(X, Y)
plt.xlabel('Number of Bikes')
plt.ylabel('temperature of the day')
plt.scatter(x,y, color='g')
plt.plot(X,y_pred, color='r')
plt.show()
print(r2_score)
