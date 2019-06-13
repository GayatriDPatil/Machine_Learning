import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import model_selection

data = pd.read_csv('/home/admin1/Desktop/Gayatri/Week2/ML_Problems/SimpleLinearRegression/bike_sharing.csv')
#print(data)
#data.head()
data.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

data = data.drop(['instant','dteday','yr'], axis=1)
data['season'] = data.season.astype('category')
data['month'] = data.month.astype('category')
data['hour'] = data.hour.astype('category')
data['holiday'] = data.holiday.astype('category')
data['weekday'] = data.weekday.astype('category')
data['workingday'] = data.workingday.astype('category')
data['weather'] = data.weather.astype('category')

data_dummy = data


def dummify_dataset(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column, drop_first=True)], axis=1)
    df = df.drop([column], axis=1)
    return df


columns_to_dummify = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
for column in columns_to_dummify:
    data_dummy = dummify_dataset(data_dummy, column)

data_dummy.head(1)

from sklearn.model_selection import train_test_split

y = data_dummy['count']
X = data_dummy.drop(['count'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

from sklearn.ensemble import RandomForestRegressor
models = [RandomForestRegressor()]


def test_algorithms(model):
    kfold = model_selection.KFold(n_splits=10, random_state=0)
    predicted = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print(predicted.mean())

reg = RandomForestRegressor()
reg.fit(X_train, y_train)
ypred = reg.predict(X_test)
print(ypred)
# Plot the residuals
residuals = y_test- ypred
fig, ax = plt.subplots()
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Number of Bikes')
ax.set_ylabel('temperature of the day')
plt.show()

r2_score = reg.score(X_test, y_test)
print(r2_score)