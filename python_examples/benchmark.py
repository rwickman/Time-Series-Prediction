from ../CryptoPrediction import CryptoPrediction
import pandas as pd
import numpy as np
from scipy import stats, special

btc_df = pd.read_csv("data/Gemini_BTCUSD_d.csv")

cp = CryptoPrediction()
data = btc_df["Close"][::-1].to_numpy()
# data, maxlog = stats.boxcox(data)
X, y = cp.create_examples(data, 14)

y_pred_naive = []
for i in range(len(X)):
    y_pred_naive.append(cp.naive_forecast(X[i]))

y_pred_avg_3 = []
for i in range(len(X)):
    y_pred_avg_3.append(cp.avg_forecast(X[i], 3))

y_pred_avg_7 = []
for i in range(len(X)):
    y_pred_avg_7.append(cp.avg_forecast(X[i], 7))

y_pred_avg_14 = []
for i in range(len(X)):
    y_pred_avg_14.append(cp.avg_forecast(X[i], 14))

y_pred_drift = []
for i in range(len(X)):
    y_pred_drift.append(cp.drift_forecast(btc_df["Close"][0:i+1].to_numpy(), 1))

y_pred_drift_partial = []
for i in range(len(X)):
    y_pred_drift_partial.append(cp.drift_forecast(X[i], 1))

# inverse_boxcox = lambda el : special.inv_boxcox(el, maxlog)
# y = np.array(list(map(inverse_boxcox, y)))

# y_pred_naive = np.array(list(map(inverse_boxcox, y_pred_naive)))
# y_pred_avg_3 = np.array(list(map(inverse_boxcox, y_pred_avg_3)))
# y_pred_avg_7 = np.array(list(map(inverse_boxcox, y_pred_avg_7)))
# y_pred_avg_14 = np.array(list(map(inverse_boxcox, y_pred_avg_14)))
# y_pred_drift = np.array(list(map(inverse_boxcox, y_pred_drift)))
# y_pred_drift_partial = np.array(list(map(inverse_boxcox, y_pred_drift_partial)))

print("NAIVE MSE: ", cp.mse(np.array(y_pred_naive), y))
print("AVERAGE MSE (3)", cp.mse(np.array(y_pred_avg_3), y))
print("AVERAGE MSE (7)", cp.mse(np.array(y_pred_avg_7), y))
print("AVERAGE MSE (14)", cp.mse(np.array(y_pred_avg_14), y))
print("DRIFT (all data) MSE: ", cp.mse(np.array(y_pred_drift), y))
print("DRIFT Partial MSE: ", cp.mse(np.array(y_pred_drift_partial), y))