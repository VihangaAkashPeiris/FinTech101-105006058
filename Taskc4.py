from Taskc2 import Loading_and_processing
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU,SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt

def make_sequences(series : pd.Series, lookback:int , lookahead:int = 1):
    values = series.values.astype(float)
    n = len(values)
    num_samples= n - lookback - lookahead + 1

    if (num_samples <= 0):
        raise ValueError (f"Not enough rows: need > lookback+lookahead-1, got {n}")
    x = np.zeros ((num_samples , lookback,1),dtype=float )
    y = np.zeros ((num_samples,),dtype=float)

    for i in range (num_samples):
        x [i,:,0] = values [i: i+ lookback]
        y [i] = values [i + lookback + lookahead - 1]

    return x,y

def build_the_model(Mtype, lookback:int,  n_layers : int, units:int , dropout:float) -> tf.keras.Model :
    model = Sequential()
    model.add(Mtype(units, return_sequences = (n_layers > 1), input_shape  =(lookback,1)))
    model.add(Dropout(dropout))


    for _ in  range (1,n_layers -1):
        model.add (Mtype(units,return_sequences =True))
        model.add (Dropout(dropout))

    if n_layers >1 :
        model.add (Mtype(units))
        model.add (Dropout(dropout))

    model.add(Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])


    return model




if __name__ == "__main__":
        # set inputs for the function.
    ticker = "CBA.AX"
    start = "2015-01-01"
    end   = "2021-01-01"
    test_ratio = 0.2
    use_scale = True
    feature_cols = ["open", "high", "low", "close", "adjclose", "volume"]

    # function which was imported from taskc2.py
    train_df, test_df, df, scalers = Loading_and_processing(
        ticker, start, end,
        split_method="ratio", test_size=test_ratio,
        scale=use_scale, feature_cols=feature_cols
    )
    # to check whether the data is loading correctly from taskc2.py to tskc3.py
    print("Train rows:", len(train_df))
    print("Test rows :", len(test_df))
    print("Full rows :", len(df))
    print(df.head())    # first 5 rows
    print(df.tail())    # last 5 rows


X_train, y_train = make_sequences(train_df["adjclose"], lookback = 60, lookahead=1)
X_test,  y_test  = make_sequences(test_df["adjclose"],  lookback = 60, lookahead=1)

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

print ("Choose a model  :\n 1:LSTM \n 2:GRU \n 3:RNN")
modeltype = int(input("Enter the model No: "))
no_layers = int (input("Enter No of Model Layers: "))
Layer_size = int (input ("Enter Layer Size: "))
no_epoches = int (input("Enter No of epoches: "))
batch = int (input("Enter batch size: "))

if modeltype == 1:
    Mtype = LSTM
elif modeltype ==2:
    Mtype = GRU
else:
    Mtype= SimpleRNN



lookback = X_train.shape[1]

model =build_the_model (Mtype, lookback=lookback,n_layers= no_layers, units=Layer_size, dropout=0.3)


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=no_epoches,
    batch_size=batch,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)
y_pred = model.predict(X_test).ravel()
plt.figure()
plt.plot(y_test, label="Actual", color = "blue")
plt.plot(y_pred, label="Predicted", color = "red")
plt.xlabel("Time (test samples)"); plt.ylabel("Price $")
plt.legend(); plt.title("Actual vs Predicted (Test)")
plt.tight_layout(); plt.show()
