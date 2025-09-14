from Taskc2 import Loading_and_processing
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU,SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt
# Make sequences to input the model. input -- > [1,2,3,4,5] [6]
def make_sequences(series : pd.Series, lookback:int , lookahead:int = 1):
    values = series.values.astype(float) # take the [adjclose] column and extact it as a numpy array and forces it to be float.
    n = len(values) #assigning the length of that numpy array to variable "n"
    num_samples= n - lookback - lookahead + 1 # this is the number of test cases that Model will perform. That means number of target values.

    if (num_samples <= 0):
        raise ValueError (f"Not enough rows: need > lookback+lookahead-1, got {n}") # if there is no target then raise an error.
    
    #the below x and y creates numpy arrays filled with zeros and it specifies the shape of the arrray.
    x = np.zeros ((num_samples , lookback,1),dtype=float )
    y = np.zeros ((num_samples,),dtype=float)

    for i in range (num_samples):
        x [i,:,0] = values [i: i+ lookback] # this fills the x array according to ith sample.
        y [i] = values [i + lookback + lookahead - 1] # This fills the target value to the Y array

    return x,y # Return two arrays with train(x) and target(y).
# Funcion to build a DL Model.
def build_the_model(Mtype, lookback:int,  n_layers : int, units:int , dropout:float) -> tf.keras.Model :
    model = Sequential() # This creates an empty model container where we can stack layers one after another
    # this add the layers to the model and specify which model and how many neurons will be there in the model and the input shape
    model.add(Mtype(units, return_sequences = (n_layers > 1), input_shape  =(lookback,1))) 
    model.add(Dropout(dropout)) # Randomly turns off some neurons to genaralize than just memorizing. (Prevent overfitting)

    # Do add the middle layers if there are more than two layers
    for _ in  range (1,n_layers -1):
        model.add (Mtype(units,return_sequences =True))
        model.add (Dropout(dropout))
    # Do add the last layer
    if n_layers >1 :
        model.add (Mtype(units))
        model.add (Dropout(dropout))

    model.add(Dense(1)) # Predict from learned patterns
    #This tells the model to learn using which optimizer, minimize MSE, and also show MAE while training.
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])


    return model # Return a model
# Function to display a summary of the model and evaluate the model
def evaluate_and_report(model, X_train, y_train, X_test, y_test, model_name="Model"):
    #Print summary (architecture)
    print(f"\n======================================  {model_name} Summary ====================================== ")
    model.summary()

    # Evaluate performance
    print(f"\n==================================  {model_name} Evaluation =======================================")
    results_train = model.evaluate(X_train, y_train, verbose=0, return_dict=True)
    results_test  = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

    print(f"Train -> Loss: {results_train['loss']:.6f}, MAE: {results_train['mae']:.6f}")
    print(f"Test  -> Loss: {results_test['loss']:.6f}, MAE: {results_test['mae']:.6f}")

    # Return results for further comparison
    return {
        "name": model_name,
        "train_loss": results_train['loss'],
        "train_mae": results_train['mae'],
        "test_loss": results_test['loss'],
        "test_mae": results_test['mae']
    }




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
    # to check whether the data is loading correctly from taskc2.py to tskc4.py
    print("Train rows:", len(train_df))
    print("Test rows :", len(test_df))
    print("Full rows :", len(df))
    print(train_df.head())    # first 5 rows
    print(test_df.tail())    # last 5 rows


X_train, y_train = make_sequences(train_df["adjclose"], lookback = 60, lookahead=1) #Make sequences from train dataframe 
X_test,  y_test  = make_sequences(test_df["adjclose"],  lookback = 60, lookahead=1) #Make sequences from test dataframe


print("Train:", X_train.shape, y_train.shape) # Print the shape of the training sequences
print("Test :", X_test.shape, y_test.shape) # Print the shape of the testing sequences

# Getting User inputs to buid the model.
print ("Choose a model  :\n 1:LSTM \n 2:GRU \n 3:RNN")
modeltype = int(input("Enter the model No: ")) # What DL model that the program should build.
no_layers = int (input("Enter No of Model Layers: ")) # number of layer should be there in the model.
Layer_size = int (input ("Enter Layer Size: ")) # this is typically how many neurons are there per a layer.
no_epoches = int (input("Enter No of epoches: ")) # This decides how many time the model go though the data.
batch = int (input("Enter batch size: ")) # How many samples should the model process in each epoch

# Based on the user input decide with model.
if modeltype == 1:
    Mtype = LSTM
    name = "LSTM"
elif modeltype ==2:
    Mtype = GRU
    name = "GRU"
else:
    Mtype= SimpleRNN
    name = "RNN"



lookback = X_train.shape[1] # This assign how many time steps to look back.
# This passes the user inputs to the model and this will create a model according to that
model =build_the_model (Mtype, lookback=lookback,
                        n_layers= no_layers,
                          units=Layer_size, dropout=0.3) 
# Early stopping: stop training if validation loss doesn't improve for 20 epochs
# and restore the model weights from the best epoch
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)
#This is the training process
history = model.fit(
    X_train, y_train, # this input features and target values. Model will learn from it.
    epochs=no_epoches, #This tells the model how many times it has to go through the data
    batch_size=batch, # Instead of training the whole data set at once it group samples.
    validation_data=(X_test, y_test), # Va;idate the predictions with test data
    callbacks=[early_stop], # Call the early_stop function that we create earlier to save time 
    verbose=1 # This controls how much info is printed while training
)
y_pred = model.predict(X_test).ravel() # Get the predicted prices.

evaluate_and_report(model, X_train, y_train, X_test, y_test, model_name=name) # evaluate the model.

# Plot Actual vs predicted prices to see how accurate the predictions are.
plt.figure()
plt.plot(y_test, label="Actual", color = "blue") # test data
plt.plot(y_pred, label="Predicted", color = "red") # predicted data
plt.xlabel("Time (test samples)"); plt.ylabel("Price $") # x axis label
plt.legend(); plt.title("Actual vs Predicted (Test)") # Legend and the title 
plt.tight_layout(); plt.show()
