import sys
sys.path.append('..')
from CryptoPrediction  import CryptoPrediction
import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#.stats.diagnostic.acorr_ljungbox

cp = CryptoPrediction()
btc_df = pd.read_csv("../data/gemini_BTCUSD_2019_1min.csv")
data = btc_df["Close"][::-1].to_numpy()
lags = 14
X, y = cp.create_examples(data, lags)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

y_pred_naive = []
for i in range(len(X)):
    y_pred_naive.append(cp.naive_forecast(X[i]))

y_pred_avg_3 = []
for i in range(len(X)):
    y_pred_avg_3.append(cp.avg_forecast(X[i], 3))


lin_reg = LinearRegression().fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)

residuals = np.subtract(y_test, np.array(y_pred_lin_reg))
res_df = pd.DataFrame(residuals)
print(res_df.describe())
print("NAIVE MSE: ", cp.mse(np.array(y_pred_naive), y))
print("AVERAGE MSE (3)", cp.mse(np.array(y_pred_avg_3), y))
print("Linear Regression MSE: ", cp.mse(y_pred_lin_reg, y_test))
# res_df.plot()
# res_df.hist()
# res_df.plot(kind='kde')
# sm.graphics.gofplots.qqplot(res_df)

sm.graphics.tsaplots.plot_acf(res_df, lags=lags)
#lbvalue, pvalue = sm.stats.diagnostic.acorr_ljungbox(res_df)
#plt.plot(pvalue[1:])

#pd.plotting.autocorrelation_plot(res_df) 
plt.show()
