import numpy as np
import tensorflow as tf
import datetime

from crypto_prediction  import CryptoPrediction
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

class CryptoLSTM:
    def __init__(self,
            lag=14,
            units=100,
            epochs=25,
            batch_size=16,
            patience=20,
            use_early_stopping=False,
            num_pred_steps=1): 
        self.lag = lag
        self.units = units
        sel.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.use_early_stopping = use_early_stopping
        self.num_pred_steps = num_pred_steps
        self.build_model()
        self.read_data()

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(LSTM(self.units, input_shape=(lags, 1)))
        self.model.add(Dense(self.num_pred_steps))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()

    def train(self):
        callbacks = [] 
        log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
        if self.use_early_stopping: 
            callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience))

        history = self.model.fit(self.X_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks)
        return history.history

    def read_data(self):
        self.X_train = np.load("data/btc_min_close_lag_{}_x_train.npy".format(self.lag))
        self.y_train = np.load("data/btc_min_close_out_{}_y_train.npy".format(self.num_pred_steps))
        
        self.X_test = np.load("data/btc_min_close_lag_{}_x_test.npy".format(self.lag))
        self.y_test = np.load("data/btc_min_close_out_{}_y_test.npy".format(self.num_pred_steps))

        self.X_val = np.load("data/btc_min_close_lag_{}_x_val.npy".format(self.lag))
        self.y_val = np.load("data/btc_min_close_out_{}_y_val.npy".format(self.num_pred_steps))

        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        self.X_val = self.X_val.reshape(X_val.shape[0], self.X_val.shape[1], 1)

