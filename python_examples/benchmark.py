import sys
sys.path.append('..')
from CryptoPrediction import CryptoPrediction
import pandas as pd
import numpy as np
from scipy import stats, special
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


cp = CryptoPrediction()
X = np.load("../data/btc_min_close_lag_14_x.npy")
y = np.load("../data/btc_min_close_lag_14_y.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

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

"""
y_pred_drift = []
for i in range(len(X)):
    y_pred_drift.append(cp.drift_forecast(btc_df["Close"][0:i+1].to_numpy(), 1))
"""

y_pred_drift_partial = []
for i in range(len(X)):
    y_pred_drift_partial.append(cp.drift_forecast(X[i], 1))

lin_reg = LinearRegression().fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
model = sm.OLS(y_train, X_train)
results  = model .fit()
print(results.summary())

lasso_reg = Lasso(alpha=0.1).fit(X_train, y_train)
y_pred_lasso_reg = lasso_reg.predict(X_test) 

# inverse_boxcox = lambda el : special.inv_boxcox(el, maxlog)
# y = np.array(list(map(inverse_boxcox, y)))

# y_pred_naive = np.array(list(map(inverse_boxcox, y_pred_naive)))
# y_pred_avg_3 = np.array(list(map(inverse_boxcox, y_pred_avg_3)))
# y_pred_avg_7 = np.array(list(map(inverse_boxcox, y_pred_avg_7)))
# y_pred_avg_14 = np.array(list(map(inverse_boxcox, y_pred_avg_14)))
# y_pred_drift = np.array(list(map(inverse_boxcox, y_pred_drift)))
# y_pred_drift_partial = np.array(list(map(inverse_boxcox, y_pred_drift_partial)))

print("NAIVE MSE: ", cp.mse(y_pred_naive, y))
print("AVERAGE MSE (3)", cp.mse(y_pred_avg_3, y))
print("AVERAGE MSE (7)", cp.mse(y_pred_avg_7, y))
print("AVERAGE MSE (14)", cp.mse(y_pred_avg_14, y))
#print("DRIFT (all data) MSE: ", cp.mse(y_pred_drift, y))
print("DRIFT Partial MSE: ", cp.mse(y_pred_drift_partial, y))
print("Linear Regression MSE: ", cp.mse(y_pred_lin_reg, y_test))
print("Lasso Regression MSE: ", cp.mse(y_pred_lasso_reg, y_test))
