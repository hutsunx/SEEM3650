import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

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
Air_Quality_Index_NOx = np.array(dataframe['Air_Quality_Index_NOx'])
Cloud_Cover = np.array(dataframe['Cloud_Cover'])
Year = np.array(dataframe['Cloud_Cover'])

Solar_radiation = np.array(dataframe['Solar_radiation'])

X = np.concatenate([Daylight.reshape(-1, 1), Cloud_Cover.reshape(-1, 1), Temperature.reshape(-1, 1), 
                    Humidity.reshape(-1, 1), Precipitation.reshape(-1, 1), Month.reshape(-1, 1), 
                    Wind_Speed.reshape(-1, 1)], axis=1)

poly6 = PolynomialFeatures(degree = 5, include_bias = False)
X_6 = poly6.fit_transform(X)

X_poly_test = poly6.fit_transform([[0.622473, 0.443623, 0.70318, 0.261116, 0, 12, 0.206549]])  # Example test data for prediction

lasso = Lasso(alpha = 0.00001)
lasso.fit(X_6, Solar_radiation)

print("If lasso, predict Solar Radiation = ", lasso.predict(X_poly_test))