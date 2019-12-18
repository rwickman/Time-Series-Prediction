import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DAYS_IN_YEAR = 365
bc_df = pd.read_csv("data/Gemini_BTCUSD_d.csv")

# Convert to datetime
bc_df["Date"] = pd.to_datetime(bc_df['Date'], errors='coerce')

# Sort years ascending
bc_df = bc_df.sort_values(by="Date")

# Group rows by year
grouped_years = bc_df.groupby(bc_df["Date"].dt.year)

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