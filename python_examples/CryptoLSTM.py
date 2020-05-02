import sys
sys.path.append('..')
from CryptoPrediction  import CryptoPrediction
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math, json
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

cp = CryptoPrediction()
lags = 14

X_train = np.load("../data/btc_min_close_lag_14_x_train.npy")
y_train = np.load("../data/btc_min_close_lag_14_y_train.npy")

X_test = np.load("../data/btc_min_close_lag_14_x_test.npy")
y_test = np.load("../data/btc_min_close_lag_14_y_test.npy")

X_val = np.load("../data/btc_min_close_lag_14_x_val.npy")
y_val = np.load("../data/btc_min_close_lag_14_y_val.npy")

stats = json.load(open("../data/btc_min_close_stats"))
btc_mean = stats["mean"]
btc_std = stats["std"]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = tf.keras.models.Sequential()
model.add(LSTM(100, input_shape=(lags, 1)))
model.add(Dense(1))
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
es_callback = EarlyStopping(monitor='val_loss', patience=25)
model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard_callback, es_callback])

results = model.evaluate(X_test, y_test)
print('test loss:', results)


# make predictions
y_train_pred = model.predict(X_train)[:, 0]
y_test_pred = model.predict(X_test)[:, 0]

# calculate mean squared error
print("BEFORE INVERSE")
trainScore = cp.mse(y_train, y_train_pred)
print("Train Score: {}  MSE".format(trainScore))
testScore = cp.mse(y_test, y_test_pred)
print("Test Score: {} MSE".format(testScore))

# Inverse standardize scaling
y_train_pred = y_train_pred * stats["std"] + stats["mean"]
y_test_pred = y_test_pred * stats["std"] + stats["mean"]
y_train = y_train * stats["std"] + stats["mean"]
y_test =  y_test * stats["std"] + stats["mean"]

# calculate mean squared error
print("AFTER INVERSE")
trainScore = cp.mse(y_train, y_train_pred)
print("Train Score: {} MSE".format(trainScore))
testScore = cp.mse(y_test, y_test_pred)
print("Test Score: {} MSE".format(testScore))
