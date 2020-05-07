import numpy as np
import datetime, json, math
from crypto_model import CryptoModel

from crypto_prediction  import CryptoPrediction
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class CryptoLinear(CryptoModel):
    def __init__(self,
            lag=16,
            num_pred_steps=1): 
        self.lag = lag
        self.num_pred_steps = num_pred_steps
        self.build()
        self.read_data()

    def build(self):
        self.model = LinearRegression()        
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)

        # Return the MSE of the validation
        return np.mean((self.model.predict(self.X_val) - self.y_val) **2)
    
    def predict(self, x):
        return self.model.predict(x)


    def residual_plots(self):
        val_preds = self.model.predict(self.X_val)
        val_mse  = mean_squared_error(val_preds, self.y_val)
        print("VAL MSE: ", val_mse)
        residuals =  val_preds - self.y_val
        residuals = np.mean(residuals, axis=1)
        val_preds_mean = np.mean(val_preds, axis=1)
        
        plt.plot(val_preds, residuals)
        plt.title("Residual Against Fitted Values")
        plt.show()
        num_sub_plots = math.ceil(self.lag / 4)

        cur_lag = 0
        for _ in range(num_sub_plots):
            fig, ax = plt.subplots(nrows=2, ncols=2)
            fig.tight_layout()
            for row in ax:
                for col in row:
                    if cur_lag >= self.lag:
                        break
                    col.title.set_text("Residual Against {} Predictor".format(cur_lag))
                    col.scatter(self.X_val[:, cur_lag], residuals)
                    cur_lag += 1
            
            #plt.show()
            plt.savefig("plots/Residual_Against_{}_Predictor".format(cur_lag))
       
        print("Making validation predictions")
        
        # Take mean of residual predictions
        residuals = np.mean(residuals, axis=1)

        print("Computing ACF Plot")
        sm.graphics.tsa.plot_acf(residuals, lags=self.lag)
        plt.show()
        plt.title("Residuals from Linear Regression model with {} lag".format(self.lag))

        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.plot(residuals)
        plt.show()
        h = np.hstack(residuals)
        plt.hist(h, bins='auto')
        plt.title("Histogram of Residuals")
        plt.show()

#lr = CryptoLinear(16, 10)
#lr.train()
#lr.residual_plots()

