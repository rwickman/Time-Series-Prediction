import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CryptoPlotting:
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