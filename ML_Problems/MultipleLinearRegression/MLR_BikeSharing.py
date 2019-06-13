import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('/home/admin1/Desktop/Gayatri/Week2/ML_Problems/SimpleLinearRegression/bike_sharing.csv')

data_count = data['cnt'].values
data_temp = data['temp'].values
data_holiday = data["holiday"].values
data_weekday = data["weekday"].values
data_humidity = data["hum"].values
data_casual = data["casual"].values
data_register = data["registered"].values

data_count = data_count.reshape(-1,1)
data_temp = data_temp.reshape(-1,1)
data_holiday = data_holiday.reshape(-1,1)
data_weekday = data_weekday.reshape(-1,1)
data_humidity = data_humidity.reshape(-1,1)
data_casual = data_casual.reshape(-1,1)
data_register = data_register.reshape(-1,1)

reg = LinearRegression()
reg.fit(data_temp, data_count)
ypredTemp = reg.predict(data_temp)
print("Temprature:", ypredTemp)
#plt.plot(data_temp, ypredTemp)
#plt.show()
#plt.scatter(data_count, ypredTemp)

reg.fit(data_holiday, data_count)
ypredholiday = reg.predict(data_holiday)
print("Holiday:", ypredholiday)
#plt.plot(data_humidity, ypredholiday)
#plt.show()

reg.fit(data_weekday, data_count)
ypredweek = reg.predict(data_weekday)
print("Weekday:", ypredweek)
#plt.plot(data_weekday, ypredweek)
#plt.show()

reg.fit(data_casual, data_count)
ypredcasual = reg.predict(data_casual)
print("Casual:", ypredcasual)
#plt.plot(data_casual, ypredcasual)
#plt.show()

reg.fit(data_register, data_count)
ypredregister = reg.predict(data_register)
print("Register:", data_register)
#plt.plot(data_register, ypredregister)
#plt.show()

reg.fit(data_humidity, data_count)
ypredhumidity = reg.predict(data_humidity)
print("Humidity:", ypredhumidity)
#plt.plot(data_humidity, ypredhumidity)
#plt.show()

