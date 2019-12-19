import sys
sys.path.append('..')
from CryptoPrediction  import CryptoPrediction
import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
#.stats.diagnostic.acorr_ljungbox

cp = CryptoPrediction()
btc_df = pd.read_csv("../data/Gemini_BTCUSD_d.csv")
data = btc_df["Close"][::-1].to_numpy()
lags = 14
X, y = cp.create_examples(data, lags)

y_pred_naive = []
for i in range(len(X)):
    y_pred_naive.append(cp.naive_forecast(X[i]))

y_pred_avg_3 = []
for i in range(len(X)):
    y_pred_avg_3.append(cp.avg_forecast(X[i], 3))

residuals = np.subtract(y, np.array(y_pred_avg_3))
res_df = pd.DataFrame(residuals)
print(res_df.describe())
print("NAIVE MSE: ", cp.mse(np.array(y_pred_naive), y))
print("AVERAGE MSE (3)", cp.mse(np.array(y_pred_avg_3), y))
# res_df.plot()
# res_df.hist()
# res_df.plot(kind='kde')
# sm.graphics.gofplots.qqplot(res_df)

sm.graphics.tsaplots.plot_acf(res_df, lags=50)
#lbvalue, pvalue = sm.stats.diagnostic.acorr_ljungbox(res_df)
#plt.plot(pvalue[1:])

#pd.plotting.autocorrelation_plot(res_df) 
plt.show()
