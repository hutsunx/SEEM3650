import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression  #import Linear Regression from python sklearn package

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

Month = np.array(dataframe['Month'])
Temperature = np.array(dataframe['Temperature'])
Humidity = np.array(dataframe['Humidity'])
Wind_Speed = np.array(dataframe['Wind_Speed'])
Precipitation = np.array(dataframe['Precipitation'])
Daylight = np.array(dataframe['Daylight'])
Cloud_Cover = np.array(dataframe['Cloud_Cover'])

Solar_radiation = np.array(dataframe['Solar_radiation'])
y = Solar_radiation

X = np.concatenate([Daylight.reshape(-1, 1), Cloud_Cover.reshape(-1, 1), Temperature.reshape(-1, 1), 
                    Humidity.reshape(-1, 1), Precipitation.reshape(-1, 1), Month.reshape(-1, 1), 
                    Wind_Speed.reshape(-1, 1)], axis=1)
# X = np.concatenate([Daylight.reshape(-1, 1), Cloud_Cover.reshape(-1, 1), Temperature.reshape(-1, 1), 
#                     Humidity.reshape(-1, 1), Precipitation.reshape(-1, 1)], axis=1)
# X = np.concatenate([Daylight.reshape(-1, 1), Cloud_Cover.reshape(-1, 1)], axis=1)

LR = LinearRegression(fit_intercept= True)
# LR = LinearRegression(fit_intercept= False)

kcv = KFold(n_splits=5, random_state=60, shuffle=True)
scores = cross_val_score(LR, X, y, scoring='neg_mean_squared_error', cv=kcv, n_jobs=1)

print(f'Linear MSE: {np.mean(np.abs(scores)):.5f}, STD: {np.std(scores):.5f}')