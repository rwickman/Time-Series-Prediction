import sys
sys.path.append('..')
from CryptoPrediction  import CryptoPrediction
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math


cp = CryptoPrediction()
btc_df = pd.read_csv("../data/gemini_BTCUSD_2019_1min.csv", usecols=["Close"])

data = btc_df["Close"][::-1].to_numpy()

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1,1))
data = np.array([v[0] for v in data])
lags = 14
X, y = cp.create_examples(data, lags)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = keras.models.Sequential()
model.add(keras.layers.LSTM(32, input_shape=(1, lags)))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
 

# make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_train = scaler.inverse_transform([y_train])
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test = scaler.inverse_transform([y_test])

# calculate root mean squared error
trainScore = mean_squared_error(y_train[0], y_train_pred[:,0])
print('Train Score: %.2f MSE' % (trainScore))
testScore = mean_squared_error(y_test[0], y_test_pred[:,0])
print('Test Score: %.2f MSE' % (testScore))

