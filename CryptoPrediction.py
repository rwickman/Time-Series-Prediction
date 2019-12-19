import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score

class CryptoPrediction:
    def __init__(self, df=None):
        self.df = df

    def mse(self, y_pred, y_true):
        return np.square(np.subtract(y_true,y_pred)).mean()
    
    def avg_forecast(self, data, t):
        """
        Return the prediction using the average of the last t observations.

        Args:
            data: A numpy array to use for forecasting
            t: the amount of lagged predictors to use.
        """
        return np.mean(data[-t:])


    def naive_forecast(self, data):
        """
        Return the prediction using the previous observations.

        Args:
            data: A numpy array to use for forecasting.
        """
        return data[-1]

    def drift_forecast(self, data, h):
        """
        Return the prediction for the future h observation using change over time.

        Args:
            data: A numpy array to use for forecasting.
            h: the number of time steps in the future to make prediction on.
        """
        if data.shape[0] == 1:
            return self.naive_forecast(data)
        return data[-1] + h * ((data[-1] - data[0]) / (data.shape[0] - 1))

    def create_examples(self, data, lag):
        X = []
        y = []
        for i in range(1, data.shape[0]):
            X.append(data[max(0, i-lag):i])
            y.append(np.array(data[i]))
        return X,y



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)

# print("X Train: ", X_train)
# print("Y Train: ", y_train)
# print("X Test: ", X_test)
# print("Y Test: ", y_test)


