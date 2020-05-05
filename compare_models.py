from crypto_lstm import CryptoLSTM
from crypto_linear import CryptoLinear
import numpy as np
import matplotlib.pyplot as plt
import argparse

class CompareModels:
    def __init__(self,
            lags,
            units,
            epochs,
            batch_sizes,
            num_iter,
            out_len,
            use_linear):
        self.lags = lags
        self.units = units
        self.epochs = epochs
        self.batch_sizes = batch_sizes
        self.num_iter = num_iter
        self.out_len = out_len
        self.use_linear = use_linear

    def compare_lag(self):
        vals = []
        for lag in self.lags:
            cur_val = 0
            for i in range(self.num_iter):
                print("Testing lag {} iter {} out of {}".format(lag, i+1, self.num_iter))
                if self.use_linear:
                    model = CryptoLinear(lag=lag, num_pred_steps=self.out_len)
                    val_mse = model.train()
                    cur_val += val_mse
                else:
                    model = CryptoLSTM(lag=lag,
                            units=self.units[0],
                            epochs=self.epochs[0],
                            batch_size=self.batch_sizes[0],
                            num_pred_steps=self.out_len)
                    history = model.train()
                    cur_val += history["val_loss"][-1]
                    print("Validation Loss: ", history["val_loss"][-1])
            vals.append(cur_val / self.num_iter)
        self.plot_loss_vs_lag(vals)

    def plot_loss_vs_lag(self, vals):
        if self.use_linear:
            plt.title("Linear Model Validation Loss vs Lag")
        else:
            plt.title("LSTM Model Validation Loss vs Lag")

        plt.xlabel("Lag")
        plt.ylabel("Validation Loss")
        plt.plot(self.lags, vals)
        plt.show()
    

def main(args):
    cm = CompareModels(args.lags,
            args.units,
            args.epochs,
            args.batch_sizes,
            args.num_iter,
            args.out_len,
            args.use_linear)
    
    if args.compare_lag:
        cm.compare_lag()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lags", nargs="*", type=int, default=[1],
            help="The lag values to compare.")
    parser.add_argument("--epochs", nargs="*", type=int, default=[1],
            help="The epoch values to compare.")
    parser.add_argument("--units", nargs="*", type=int, default=[100],
            help="The unit values to compare.")
    parser.add_argument("--batch_sizes", nargs="*", type=int, default=[256],
            help="The epoch values to compare.")
    parser.add_argument("--compare_lag", action="store_true",
            help="The epoch values to compare.")
    parser.add_argument("--num_iter", default=1, type=int,
            help="The number of iterations to test the same value.")
    parser.add_argument("--out_len", default=1, type=int,
            help="The number of timesteps to forecast.")
    parsed.add_argument("--use_linear", action="store_true",
            help="Test linear regression")
    main(parser.parse_args())
