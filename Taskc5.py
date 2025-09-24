from Taskc2 import Loading_and_processing
from Taskc4 import make_sequences
from Taskc4 import build_the_model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU,SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt



# Make sequences to input the model. input -- > [1,2,3,4,5] [6]
def make_sequences_multi_step(series : pd.Series, lookback:int , lookahead:int =5):
    values = series.values.astype(float) # take the [adjclose] column and extact it as a numpy array and forces it to be float.
    n = len(values) #assigning the length of that numpy array to variable "n"
    num_samples= n - lookback - lookahead + 1 # this is the number of test cases that Model will perform. That means number of target values.

    if (num_samples <= 0):
        raise ValueError (f"Not enough rows: need > lookback+lookahead-1, got {n}") # if there is no target then raise an error.
    
    #the below x and y creates numpy arrays filled with zeros and it specifies the shape of the arrray.
    x = np.zeros ((num_samples , lookback,1),dtype=float )
    y = np.zeros ((num_samples,lookahead),dtype=float)

    for i in range (num_samples):
        x [i,:,0] = values [i: i+ lookback] # this fills the x array according to ith sample.
        y [i,:] = values [i + lookback: lookback +lookahead + i ] # This fills the target value to the Y array

    return x,y # Return two arrays with train(x) and target(y).
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

    print("Train rows:", len(train_df))
    print("Test rows :", len(test_df))
    print("Full rows :", len(df))
    print(train_df.head())
    print(test_df.tail())

    # Sequences
    X_train, y_train = make_sequences_multi_step(train_df["adjclose"], lookback=60, lookahead=5)
    X_test,  y_test  = make_sequences_multi_step(test_df["adjclose"], lookback=60, lookahead=5)

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # Getting User inputs to build the model
    print("Choose a model  :\n 1:LSTM \n 2:GRU \n 3:RNN")
    modeltype = int(input("Enter the model No: "))
    no_layers = int(input("Enter No of Model Layers: "))
    Layer_size = int(input("Enter Layer Size: "))
    no_epoches = int(input("Enter No of epoches: "))
    batch = int(input("Enter batch size: "))

    if modeltype == 1:
        Mtype = LSTM
        name = "LSTM"
    elif modeltype == 2:
        Mtype = GRU
        name = "GRU"
    else:
        Mtype = SimpleRNN
        name = "RNN"

    lookback = X_train.shape[1]
    model = build_the_model(Mtype, lookback=lookback,
                            n_layers=no_layers,
                            units=Layer_size, dropout=0.3)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=no_epoches,
        batch_size=batch,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test)

    # Plot
    plt.figure()
    plt.plot(y_test, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="red")
    plt.xlabel("Time (test samples)"); plt.ylabel("Price $")
    plt.legend(); plt.title("Actual vs Predicted (Test)")
    plt.tight_layout(); plt.show()


    horizon = y_test.shape[1]
    data = {"Sample": range(len(y_test))}

    for h in range(horizon):
        data[f"t+{h+1}_Actual"] = y_test[:, h]
        data[f"t+{h+1}_Pred"]   = y_pred[:, h]
        data[f" "] = [""] * len(y_test)   # spacer column

    df_summary = pd.DataFrame(data)

# save to CSV with better spacing
    df_summary.to_csv("multistep_predictions_summary.csv", index=False)

    print("Summary saved with spacing columns for readability")