import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score

class CryptoPrediction:
    def __init__(self, df=None):
        self.df = df
        
    def group_by_year(self, df):
        # Convert to datetime
        df["Date"] = pd.to_datetime(df['Date'], errors='coerce')
        # Sort years ascending
        df = df.sort_values(by="Date")    
        # Group rows by year
        return df.groupby(df["Date"].dt.year)

    def seasonal_plot(self):
        self.group_by_year(self.df)
        # Plot the Seasonal plot
        for group in grouped_years:
            price_arr = group[1]["Close"].to_numpy()
            if price_arr.shape[0] < DAYS_IN_YEAR:
                price_arr = np.pad(price_arr, (DAYS_IN_YEAR - price_arr.shape[0], 0))
            plt.plot(price_arr, label=group[0])
            #plt.plot("x", str(group[0]), data=group[1]["Close"].to_numpy())

        plt.title("Seasonal Plot")
        plt.xlabel("Day")
        plt.ylabel("Price")

        plt.legend()
        plt.show()

    def plot_acf(self):
        sm.graphics.tsa.plot_acf(self.df["Close"][::-1])
        plt.show()
    
    def plot_pacf_(self):
        sm.graphics.tsa.plot_acf(self.df["Close"][::-1])
        plt.show()

    def mse(self, y_pred, y_true):
        return np.square(np.subtract(y_true,y_pred)).mean()
    
    def avg_forecast(self, data, t):
        """
        Return the prediction using the average of the last t observations.

        Args:
            data: A numpy array to use for forecasting
            t: the amount of lagged predictors to use.
        """
        return numpy.mean(data[-t:])


    def naive_forecast(self, data):
        """
        Return the prediction using the previous observations.

        Args:
            data: A numpy array to use for forecasting.
        """
        return data[-1]

    def drift_forecast(self, data, h):
        """
        Return the prediction for the future h observation using change over time.

        Args:
            data: A numpy array to use for forecasting.
            h: the number of time steps in the future to make prediction on.
        """
        return data[-1] + h * ((data[-1] - data[0]) / (data.shape[0] - 1))

    def create_examples(self, data, lag):
        X = []
        y = []
        for i in range(1, data.shape[0]):
            X.append(data[max(0, i-lag):i])
            y.append(np.array(data[i]))
        return X,y

btc_df = pd.read_csv("data/Gemini_BTCUSD_d.csv")

cp = CryptoPrediction()
X, y = cp.create_examples(btc_df["Close"][0:100].to_numpy(), 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)
# print("X Train: ", X_train)
# print("Y Train: ", y_train)
# print("X Test: ", X_test)
# print("Y Test: ", y_test)


