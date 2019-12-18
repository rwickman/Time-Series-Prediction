import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def group_by_year(df):
    # Convert to datetime
    df["Date"] = pd.to_datetime(df['Date'], errors='coerce')
    # Sort years ascending
    df = df.sort_values(by="Date")    
    # Group rows by year
    return df.groupby(df["Date"].dt.year)


btc_df = pd.read_csv("data/Gemini_BTCUSD_d.csv")
eth_df = pd.read_csv("data/Gemini_ETHUSD_d.csv")
ltc_df = pd.read_csv("data/Gemini_LTCUSD_d.csv")

btc_grouped = group_by_year(btc_df)
eth_grouped = group_by_year(eth_df)
ltc_grouped = group_by_year(ltc_df)


#np.corrcoef(eth_grouped.get_group(2015)["Close"].to_numpy() ,btc_grouped.get_group(2015)["Close"].to_numpy())
#plt.scatter(eth_grouped.get_group(2018)["Close"].to_numpy(), btc_grouped.get_group(2018)["Close"].to_numpy())
# plt.scatter(ltc_grouped.get_group(2019)["Close"].to_numpy(), btc_grouped.get_group(2019)["Close"].to_numpy())

all_df = df = pd.DataFrame({"btc" : btc_grouped.get_group(2019)["Close"].to_numpy(), "eth" : eth_grouped.get_group(2019)["Close"].to_numpy(), "ltc" : ltc_grouped.get_group(2019)["Close"].to_numpy()})
print(df.corr())
scatter_matrix(all_df, figsize=(6, 6))
plt.matshow(all_df.corr())
plt.xticks(range(len(all_df.columns)), all_df.columns)
plt.yticks(range(len(all_df.columns)), all_df.columns)
plt.colorbar()
plt.show()