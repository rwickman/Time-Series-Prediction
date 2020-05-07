import tensorflow as tf
import datetime
from crypto_model import CryptoModel

from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam

class CryptoMLP(CryptoModel):
    def __init__(self,
            lag=14,
            units=64,
            epochs=25,
            batch_size=16,
            patience=20,
            use_early_stopping=False,
            num_pred_steps=1,
            num_hidden=2): 
        super().__init__(lag, num_pred_steps)
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.use_early_stopping = use_early_stopping
        self.num_hidden = num_hidden
        self.build_model()
        self.read_data()

    def build_model(self):
        #self.model = tf.keras.models.Sequential()
        inputs = Input(shape=(self.lag, ))
        x = Dense(self.units, activation="relu")(inputs)
        #x = BatchNormalization()(x)
        for i in range(self.num_hidden - 1):
            x = Dense(self.units, activation="relu")(x)
            #if i+1 != self.num_hidden - 1:
            #    x = BatchNormalization()(x)
        
        out_pred = Dense(self.num_pred_steps)(x)
        self.model = Model(inputs=inputs, outputs=out_pred)
        self.model.compile(loss='mean_squared_error', optimizer="adam")
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
        self.model.save("saved_models/mlp")

    def load(self):
        self.model = load_model("saved_models/mlp")

mlp_model = CryptoMLP(epochs=20, lag=16, batch_size=256, units=128, num_hidden=3, num_pred_steps=10)
mlp_model.load()
#print(mlp_model.train())
mlp_model.plot_val_one_step()
mlp_model.plot_val_pred()
mlp_model.save()
