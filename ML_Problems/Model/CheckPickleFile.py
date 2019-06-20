from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#plt.rcParams['figure.figsize'] = (20.0, 10.0)
#Reading data
data = pd.read_csv('/home/admin1/Desktop/Gayatri/Week2/ML_Problems/SimpleLinearRegression/Salary_Data.csv')
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

pickleread = open("PickleFileTest","rb")
list  = pickle.load(pickleread)


plt.scatter(X, Y, color="r")
plt.plot(list[0],list[1], color='b')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()