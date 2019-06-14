import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# split dataset
from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
y_test = sc.transform(x_test)

#Fitting classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

#Predecting the test set result
ypred = classifier.predict(x_test)
print(ypred)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, ypred)

#Visualizing the trainig set result
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:,0].max()+1, step= 0.01),
                     np.arange(start= x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step= 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x2.max())
plt.ylim(x1.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set== j, 0], x_set[y_set== j, 1],
                c = ListedColormap(('red','green'))(i), label=j)
plt.title('Decision Tree(Training set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'''#Visualizing the Test set result
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:,0].max()+1, step= 0.01),
                     np.arange(start= x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step= 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x2.max())
plt.ylim(x1.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set== j, 0], x_set[y_set== j, 1],
                c = ListedColormap(('red','green'))(i), label=j)
plt.title('Decision Tree(Test set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()'''

