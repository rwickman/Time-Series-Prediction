import numpy as np
import pandas as pd

csv_file = "../data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"

btc_df = pd.read_csv(csv_file)
btc_df = btc_df.dropna()


btc_arr = btc_df["Close"].to_numpy()
np.save("btc_min_close", btc_arr)
