#Importing the libarires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('/home/admin1/Desktop/Gayatri/Week2/ML_Problems/Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN' , strategy='mean', axis=0)
imputer.fit(x[:, 1:2])
x[:, 1:2] = imputer.transform(x[:, 1:2] )

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])  # 0 is the column for encoder keep into first column
onehotencoder = OneHotEncoder(categorical_features= [0])
x = OneHotEncoder.fit_transform(x).toarray()

#Taking care of missisng data
#ifelse use function

#Encoding categorical data
#dataset$column = factor(dataset$column,
 #                       levels = c('','',''),
  #                      labels = c(1,2,3))

#splitinmg the data into trainiig and test
from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2#half data 50%,good size is 20%,random_state =0)
#libarary(caTools)ga

#Feature scaling - we can scal

#Steps
#1)Get the dataset
#2) Importing librairies
#3) Take care of missing dataset
#4) Encode categorical Data
#5) split the data set into training set and test set
#6)Feature scaling
#7)Data processing template