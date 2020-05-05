import numpy as np
import pandas as pd

from crypto_prediction import CryptoPrediction
from sklearn.model_selection import train_test_split
import argparse, json, os


class CreateData:
    @staticmethod
    def create_examples(data, lag, out_len):
        X = []
        y = []
        for i in range(1, data.shape[0]):
            x_example = data[max(0, i-lag):i]
            X.append(np.pad(x_example, (lag - x_example.shape[0], 0)))
            y_example = data[i:i+out_len]
            y.append(np.pad(y_example, (0, out_len - y_example.shape[0])))
        return np.array(X), np.array(y)
    
    @staticmethod
    def create(lag, csv_file, out_len):
        """
        Create dataset for training, testing, and validation.

        Args:
            lag: a integer for the number of timesteps for the input
            csv_file: a string of the csv data filename
            out_len: a integer giving the forecast length
        """
        btc_df = pd.read_csv(csv_file)
        btc_df = btc_df.dropna()
        btc_data = btc_df["Close"].to_numpy()
        
        btc_min_close_filename = "data/btc_min_close.npy" 
        
        if not os.path.exists(btc_min_close_filename):
            np.save(btc_min_close_filename, btc_data)
        
        # Standardize the data
        btc_mean = btc_data.mean()
        btc_std = btc_data.std()
        btc_data = (btc_data - btc_mean) / btc_std

        # Create dataset
        print("Creating examples")
        data = CreateData.create_examples(btc_data, lag, out_len)
        
        print("Splitting and saving")
        # Create 60-20-20 split
        # Split training and testing
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.20, shuffle=False)
        # Split training and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

        # Save the dataset
        x_file = "data/btc_min_close_lag_{}_x".format(lag)
        y_file ="data/btc_min_close_out_{}_y".format(out_len)  

        x_train_file = "data/btc_min_close_lag_{}_x_train".format(lag)
        y_train_file ="data/btc_min_close_out_{}_y_train".format(out_len)

        x_test_file = "data/btc_min_close_lag_{}_x_test".format(lag)
        y_test_file ="data/btc_min_close_out_{}_y_test".format(out_len)

        x_val_file = "data/btc_min_close_lag_{}_x_val".format(lag)
        y_val_file ="data/btc_min_close_out_{}_y_val".format(out_len)

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


def main(args):
    for lag in args.lags:
        CreateData.create(lag, args.csv_file, args.out_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lags", nargs="*", default=[], type=int,
            help="The lags for the input.")
    parser.add_argument("--out_len", type=int, default=1,
            help="The number of timesteps to forecast.")
    parser.add_argument("--csv_file",
            help="The data csv file.")
    main(parser.parse_args())
