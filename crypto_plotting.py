import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class CryptoPlotting:
    def __init__(self):
        self.data = np.load("data/btc_min_close.npy")

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
        sm.graphics.tsa.plot_acf(self.data[-200000:], lags=240)
        plt.show()
    
    def plot_pacf(self):
        sm.graphics.tsa.plot_pacf(self.data[-500000:], lags=60, zero=False)
        plt.show()

plotter = CryptoPlotting()
plotter.plot_pacf()
