import sys
sys.path.append('..')
from CryptoPrediction  import CryptoPrediction
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math

cp = CryptoPrediction()
btc_df = pd.read_csv("../data/gemini_BTCUSD_2019_1min.csv")
data = btc_df["Close"][::-1].to_numpy()
# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
lags = 14
X, y = cp.create_examples(data, lags)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(16, input_shape=(1, lags)))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([y_test])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

