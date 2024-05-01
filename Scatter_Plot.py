import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_types = {
    'Year': float,
    'Month': float,
    'Day': float,
    'Temperature': float,
    'Humidity': float,
    'Wind_Speed': float,
    'Precipitation': float,
    'Daylight': float,
    'Air_Quality_Index_NOx': float,
    'Cloud_Cover': float,
    'Solar_radiation': float,
}

dataframe = pd.read_csv('SEEM3650_Project_Data_Set_preprocessed.csv', dtype = data_types)

Daylight = np.array(dataframe['Daylight'])
# Cloud_Cover = np.array(dataframe['Cloud_Cover'])
# Temperature = np.array(dataframe['Temperature'])
# Humidity = np.array(dataframe['Humidity'])
# Precipitation = np.array(dataframe['Precipitation'])
# Month = np.array(dataframe['Month'])
# Wind_Speed = np.array(dataframe['Wind_Speed'])

Solar_radiation = np.array(dataframe['Solar_radiation'])

plt.scatter(Daylight, Solar_radiation, s=20, label="Samples") 
# plt.scatter(Cloud_Cover, Solar_radiation, s=20, label="Samples") 
# plt.scatter(Temperature, Solar_radiation, s=20, label="Samples") 
# plt.scatter(Humidity, Solar_radiation, s=20, label="Samples") 
# plt.scatter(Precipitation, Solar_radiation, s=20, label="Samples") 
# plt.scatter(Month, Solar_radiation, s=20, label="Samples") 
# plt.scatter(Wind_Speed, Solar_radiation, s=20, label="Samples") 

plt.xlabel("Daylight")
# plt.xlabel("Cloud_Cover")
# plt.xlabel("Temperature")
# plt.xlabel("Humidity")
# plt.xlabel("Precipitation")
# plt.xlabel("Month")
# plt.xlabel("Wind_Speed")

plt.ylabel("Solar_radiation")
plt.legend()
plt.show()