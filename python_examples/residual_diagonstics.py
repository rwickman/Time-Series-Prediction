from ../CryptoPrediction import CryptoPrediction
import pandas as pd
import numpy as np
import statsmodels as sm
#.stats.diagnostic.acorr_ljungbox
data = btc_df["Close"][::-1].to_numpy()
lags = 14
X, y = cp.create_examples(data, lags)

y_pred_naive = []
for i in range(len(X)):
    y_pred_naive.append(cp.naive_forecast(X[i]))

print("NAIVE MSE: ", cp.mse(np.array(y_pred_naive), y))