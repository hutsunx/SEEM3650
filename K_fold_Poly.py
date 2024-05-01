import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
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
Cloud_Cover = np.array(dataframe['Cloud_Cover'])

Solar_radiation = np.array(dataframe['Solar_radiation'])
y = Solar_radiation

X = np.concatenate([Daylight.reshape(-1, 1), Cloud_Cover.reshape(-1, 1), Temperature.reshape(-1, 1), 
                    Humidity.reshape(-1, 1), Precipitation.reshape(-1, 1), Month.reshape(-1, 1), 
                    Wind_Speed.reshape(-1, 1)], axis=1)
# X = np.concatenate([Daylight.reshape(-1, 1), Cloud_Cover.reshape(-1, 1), Temperature.reshape(-1, 1), 
#                     Humidity.reshape(-1, 1), Precipitation.reshape(-1, 1)], axis=1)
# X = np.concatenate([Daylight.reshape(-1, 1), Cloud_Cover.reshape(-1, 1)], axis=1)

DG_MSE = []
DG_MSE_alpha = []

for degree in range(5):

  poly = PolynomialFeatures(degree = degree + 1, include_bias = False)
  X_poly = poly.fit_transform(X)

  alpha_vect = np.array([0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
  kcv = KFold(n_splits = 5, random_state = 60, shuffle = True)
  CV_MSE = []

  for alpha in alpha_vect:
    # rr = Ridge(alpha = alpha)
    lasso = Lasso(alpha = alpha)
    scores = cross_val_score(lasso, X_poly, y, scoring = 'neg_mean_squared_error', cv = kcv, n_jobs = 1)
    CV_MSE.append(np.mean(np.abs(scores)))

    print(f'degree = {degree + 1}, alpha = {alpha}, Polynomial MSE: {np.mean(np.abs(scores)):.5f}, STD: {np.std(scores):.5f}')

  DG_MSE_alpha.append(alpha_vect[CV_MSE.index(min(CV_MSE))])
  DG_MSE.append(min(CV_MSE))


print(f"Best Alpha in each degree = {DG_MSE_alpha}")
print(f"Best MSE in each degree = {DG_MSE}")