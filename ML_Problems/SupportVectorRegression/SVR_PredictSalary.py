import numpy as np
import  matplotlib.pyplot as plt
import  pandas as pd

#importing the dataset
dataset = pd.read_csv('/home/admin1/Desktop/Gayatri/Week2/ML_Problems/DecisionTreeRegression/Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#fitting svr to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

#predicting a new result
ypred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

#Visualizing the svr result
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title(SVR)
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()