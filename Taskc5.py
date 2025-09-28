from Taskc2 import Loading_and_processing
from Taskc4 import build_the_model
from Taskc4 import evaluate_and_report
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU,SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt
import joblib


#===================================MULTISTEP___FUNCTION==============================================
# Make sequences to input the model. input -- > if k value = 3 then [1,2,3,4,5] [6,7,8]
def make_sequences_multi_step(series : pd.Series, lookback:int , lookahead:int ):
    values = series.values.astype(float) # take the [adjclose] column and extact it as a numpy array and forces it to be float.
    n = len(values) #assigning the length of that numpy array to variable "n"
    num_samples= n - lookback - lookahead + 1 # this is the number of test cases that Model will perform. That means number of target values.

    if (num_samples <= 0):
        raise ValueError (f"Not enough rows: need > lookback+lookahead-1, got {n}") # if there is no target then raise an error.
    
    #the below x and y creates numpy arrays filled with zeros and it specifies the shape of the arrray.
    x = np.zeros ((num_samples , lookback,1),dtype=float )
    y = np.zeros ((num_samples,lookahead),dtype=float) #since we get predictions for k value, now y is a 2d array shaped like (rows,columns) 

    for i in range (num_samples):
        x [i,:,0] = values [i: i+ lookback] # this fills the x array according to ith sample.
        y [i,:] = values [i + lookback: lookback +lookahead + i ] # This fills the target value to the Y array According to k value.

    return x,y # Return two arrays with train(x) and target(y).

#======================================MULTIVARIATE_FUNCTION===========================================

def make_sequences_multivariate(df:pd.DataFrame, lookback:int , lookahead:int = 1, targeted_col : int=0):
    values = df.values.astype(float) # take the dataframe instead of gettinga seiris of data and extract it as a numpy array and forces it to be float.
    n, n_features = values.shape #assigning the shape of the values to n and n_features variable. Since it is a 2d array the shape is [rows,columns].
    num_samples= n - lookback - lookahead + 1 # this is the number of test cases that Model will perform. That means number of target values.

    if (num_samples <= 0):
        raise ValueError (f"Not enough rows: need > lookback+lookahead-1, got {n}") # if there is no target then raise an error.
    
    #the below x     and y creates numpy arrays filled with zeros and it specifies the shape of the arrray.
    x = np.zeros ((num_samples , lookback,n_features),dtype=float ) # since this is a 3d array we have set the shape (rows,timesteps,columns)
    y = np.zeros ((num_samples,),dtype=float) #this is a 1d array shape is just the length (rows,)

    for i in range (num_samples):
        x [i] = values [i: i+ lookback] # this fills the x array according to ith sample.
        y [i] = values [i + lookback + lookahead - 1,targeted_col] # This fills the target value to the Y array

    return x,y # Return two arrays with train(x) and target(y).


#======================================MULTISTEP+MULTIVARIATE============================================
def multistep_and_multivariate (df:pd.DataFrame, lookback:int, lookahead:int, targeted_col:int = 0):
    values =  df.values.astype(float) # get the dataframe instead of taking a series and force the type of the data to float.
    n,n_features = values.shape # Assigning the shape of the "values" to n and n_features. N gets the length and n_features get how many columns are there.
    num_samples = n - lookback- lookahead + 1 # the number of test cases
    
    if (num_samples <= 0):
        raise ValueError (f"Not enough rows: need > lookback+lookahead-1, got {n}") # if there is no target then raise an error.
    
    x = np.zeros((num_samples,lookback,n_features),dtype=float)  # since this is a 3d array we have set the shape (rows,timesteps,columns)
    y = np.zeros((num_samples,lookahead)) #since we get predictions for k value, now y is a 2d array shaped like (rows,columns) 
    for i in range (num_samples):
        x[i] = values [i: i+lookback, :] # this fills the x array according to ith sample.
        y[i,:] = values [i + lookback: lookback +lookahead + i , targeted_col] # This fills the target value to the Y array

    
    return x,y # Return two arrays with train(x) and target(y).

if __name__ == "__main__":
    # set inputs for the function.
    ticker = "CBA.AX"
    start = "2020-01-01"
    end   = "2025-01-01"
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

    kvalue =  int(input (" How many days would you prefer to predict ?"))

    # Sequences multistep
    #X_train, y_train = make_sequences_multi_step(train_df["adjclose"], lookback=60, lookahead=kvalue)
    #X_test,  y_test  = make_sequences_multi_step(test_df["adjclose"], lookback=60, lookahead=kvalue)

    # Sequences multivariate
   #X_train, y_train = make_sequences_multivariate(train_df[['adjclose', 'volume', 'open','close', 'high', 'low']], lookback=60, lookahead=1)
    #X_test,  y_test  = make_sequences_multivariate(test_df[['adjclose', 'volume', 'open','close', 'high', 'low']], lookback=60, lookahead=1)

    X_train, y_train = multistep_and_multivariate(train_df[['adjclose', 'volume', 'open','close', 'high', 'low']], lookback=60, lookahead=kvalue)
    X_test,  y_test  = multistep_and_multivariate(test_df[['adjclose', 'volume', 'open','close', 'high', 'low']], lookback=60, lookahead=kvalue)
    
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
                            units=Layer_size, dropout=0.3, dense_layers=kvalue)

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

    scalers = joblib.load("feature_scalers.pkl") # loaded the scaler dictionary we saved in Taskc02.py
    t_scaler = scalers["adjclose"]        # wanted to inverse the adjclose price, So from the dictionary select the correct scaler.

    N, K = y_test.shape                    # (num_samples, lookahead)
    y_test_inv = t_scaler.inverse_transform(y_test.reshape(-1,1)).reshape(N, K) # inverse the y_test prices to its original price
    y_pred_inv = t_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(N, K) #inverse the y_pred prices to its original prices
    
    evaluate_and_report(model, X_train, y_train, X_test, y_test, model_name=name)
    for i in range(kvalue): 
    # Plot
        plt.figure()
        plt.plot(y_test[:,i], label=f" Actual_{i+1}", color="blue")
        plt.plot(y_pred[:,i], label=f"Predicted_{i+1}", color="red")
        plt.xlabel("Time (test samples)"); plt.ylabel("Price $")
        plt.legend(); 
        plt.title(f"Actual_{i+1} vs Predicted_{i+1} (Test)")
        plt.tight_layout(); 
        plt.savefig(f"Taskc5-plots/my_plot_{i+1}_{name}.jpg", dpi=300) 
        plt.show()

#============================================save predictions into a csv summary file==========================================
    horizon = y_test.shape[1] # lookback
    data = {"Sample": range(len(y_test))} #create the index column named as "Sample"

    for h in range(horizon):
        data[f"{h+1}_Actual"] = y_test_inv[:, h] # Creates actual prices column
        data[f"{h+1}_Pred"]   = y_pred_inv[:, h] # Creates predicted prices column
        

    df_summary = pd.DataFrame(data)

    # save results to a csv.
    df_summary.to_csv("multistep_predictions_summary.csv", index=False) # Convert to CSV.

    print("Summary saved to multistep_predictions_summary.csv")