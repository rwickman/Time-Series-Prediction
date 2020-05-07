import numpy as np
import datetime
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class CryptoModel(ABC):
    def __init__(self,
            lag=16,
            num_pred_steps=1):
        self.lag = lag
        self.num_pred_steps = num_pred_steps
    
    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    def val_mse(self):
        return mean_squared_error(self.predict(self.X_val), self.y_val)

    def test_mse(self):
        return mean_squared_error(self.predict(self.X_test), self.y_test)
    
    def read_data(self, reshape=False):
        self.X_train = np.load("data/btc_min_close_lag_{}_x_train.npy".format(self.lag))
        self.y_train = np.load("data/btc_min_close_out_{}_y_train.npy".format(self.num_pred_steps))
        
        self.X_test = np.load("data/btc_min_close_lag_{}_x_test.npy".format(self.lag))
        self.y_test = np.load("data/btc_min_close_out_{}_y_test.npy".format(self.num_pred_steps))

        self.X_val = np.load("data/btc_min_close_lag_{}_x_val.npy".format(self.lag))
        self.y_val = np.load("data/btc_min_close_out_{}_y_val.npy".format(self.num_pred_steps))
        if reshape:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
            self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
            self.X_val = self.X_val.reshape(self.X_val.shape[0], self.X_val.shape[1], 1)

    def plot_val_pred(self, num_steps=100, time_step=0):
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.plot(self.y_val[:, time_step][:num_steps])
        pred = self.model.predict(self.X_val)
        plt.plot(pred[:, time_step][:num_steps])
        plt.legend(["Actual", "Forecast"])
        plt.show()

    def plot_val_one_step(self, time_step=0):
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.plot(self.y_val[:, time_step][:self.num_pred_steps])
        pred = self.model.predict([self.X_val[:time_step+1]])
        plt.plot(pred[time_step])
        plt.legend(["Actual", "Forecast"])
        plt.show()



