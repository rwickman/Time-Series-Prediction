import numpy as np
import tensorflow as tf
import datetime

from crypto_prediction  import CryptoPrediction
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle

class CryptoLinear:
    def __init__(self,
            lag=14,
            num_pred_steps=1): 
        self.lag = lag
        self.num_pred_steps = num_pred_steps
        self.build_model()
        self.read_data()

    def build_model(self):
        self.model = LinearRegression()        
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)

        # Return the MSE of the validation
        return np.mean((self.model.predict(self.X_val) - self.y_val) **2)

    def read_data(self):
        self.X_train = np.load("data/btc_min_close_lag_{}_x_train.npy".format(self.lag))
        self.y_train = np.load("data/btc_min_close_out_{}_y_train.npy".format(self.num_pred_steps))
        
        self.X_test = np.load("data/btc_min_close_lag_{}_x_test.npy".format(self.lag))
        self.y_test = np.load("data/btc_min_close_out_{}_y_test.npy".format(self.num_pred_steps))

        self.X_val = np.load("data/btc_min_close_lag_{}_x_val.npy".format(self.lag))
        self.y_val = np.load("data/btc_min_close_out_{}_y_val.npy".format(self.num_pred_steps))
