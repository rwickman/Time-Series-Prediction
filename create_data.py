import numpy as np
from CryptoPrediction import CryptoPrediction
from sklearn.model_selection import train_test_split
import json

lag = 14
cp = CryptoPrediction()

# Load data
btc_min_datafile= "data/btc_min_close.npy"
btc_data = np.load(btc_min_datafile)

# Standardize the data
btc_mean = btc_data.mean()
btc_std = btc_data.std()
btc_data = (btc_data - btc_mean) / btc_std

# Create dataset
data = cp.create_examples(btc_data, lag)

X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.20, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=True)

# Save the dataset
x_file = "data/btc_min_close_lag_{}_x".format(lag)
y_file ="data/btc_min_close_lag_{}_y".format(lag)  

x_train_file = "data/btc_min_close_lag_{}_x_train".format(lag)
y_train_file ="data/btc_min_close_lag_{}_y_train".format(lag)  

x_test_file = "data/btc_min_close_lag_{}_x_test".format(lag)
y_test_file ="data/btc_min_close_lag_{}_y_test".format(lag)  

x_val_file = "data/btc_min_close_lag_{}_x_val".format(lag)
y_val_file ="data/btc_min_close_lag_{}_y_val".format(lag)  

stats_datafile = "data/btc_min_close_stats.json"


# Save the numpy data
np.save(x_file, data[0])
np.save(y_file, data[1])

np.save(x_train_file, X_train)
np.save(y_train_file, y_train)

np.save(x_test_file, X_test)
np.save(y_test_file, y_test)

np.save(x_val_file, X_val)
np.save(y_val_file, y_val)

stats = {"mean" : btc_mean, "std" : btc_std}
json.dump(stats, open(stats_datafile, "w"))

