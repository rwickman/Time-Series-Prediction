import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class CryptoPrediction:
    def __init__(self, df):
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
            data: A numpy array to make a prediction on.
            t: the amount of lagged predictors to use.
        """
        return numpy.mean(data[-t:])


    def naive_forecast(self, data):
        """
        Return the prediction using the previous observations.

        Args:
            data: A numpy array to make a prediction on.
        """
        return data[-1]
