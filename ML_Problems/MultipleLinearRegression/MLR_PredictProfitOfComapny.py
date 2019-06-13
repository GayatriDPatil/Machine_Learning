import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data=pd.read_csv('50_Startups.csv')
print(data)
x= data.iloc[:, :-1]
y= data.iloc[:, 4]

states = pd.get_dummies(x['State'],drop_first=True)
X = x.drop('State', axis= 1)
X = pd.concat([X,states],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_Pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_test,y_Pred)

print(score)
plt.xlabel('Expenses')
plt.ylabel('Profit')
#plt.plot(y_test,y_Pred)
#plt.show()