import tensorflow as tf
import datetime
from crypto_model import CryptoModel

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

class CryptoLSTM(CryptoModel):
    def __init__(self,
            lag=14,
            units=100,
            epochs=25,
            batch_size=16,
            patience=20,
            use_early_stopping=False,
            num_pred_steps=1): 
        super().__init__(lag, num_pred_steps)
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.use_early_stopping = use_early_stopping
        self.build_model()
        self.read_data(reshape=True)

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(LSTM(self.units, input_shape=(self.lag, 1)))
        self.model.add(Dense(self.num_pred_steps))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()

    def train(self):
        callbacks = [] 
        log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
        if self.use_early_stopping: 
            callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience))

        history = self.model.fit(self.X_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks,
                shuffle=True)
        return history.history

    def predict(self, x):
        return self.model.predict(x)

    def save(self):
        self.model.save("saved_models/lstm")

    def load(self):
        self.model = load_model("saved_models/lstm")


model = CryptoLSTM(lag=16, num_pred_steps=10, epochs=20, batch_size=256)
print(model.train())
model.plot_val_pred()
model.save()
