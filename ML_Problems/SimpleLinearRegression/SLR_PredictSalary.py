#%matplotlib inline

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plt.rcParams['figure.figsize'] = (20.0, 10.0)
#Reading data
data = pd.read_csv('Salary_Data.csv')
print (data)
data.head()

#collecting x and y
x = data['YearsExperience'].values
y = data['Salary'].values
#mean x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

#total no of values
m = len(x)

#using the formula to calculate b1 and b2
numer = 0
denom = 0
for i in range(m):
    numer +=(x[i] - mean_x) * (y[i] - mean_y)
    denom +=(x[i] - mean_x) ** 2
b1 = numer/denom
b0 = mean_y - (b1 * mean_x)
#print coefficients
print(b1,b0)

#ploting values and Regression line
max_x = np.max(x + 100)
min_x = np.min(x - 100)

# calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 *x

#ploting line
plt.plot(x,y,color='#58b970', label='Regression Line')
#ploting scatter points
plt.scatter(x,y,c='#ef5423', label='Scatter Plot')

plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.legend()
plt.show()

#calculate r2 value
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 +b1*x
    ss_t += (y[i] - mean_y) **2
    ss_r += (y[i] - y_pred) **2
r2 = 1 - (ss_r/ss_t)
print(r2)