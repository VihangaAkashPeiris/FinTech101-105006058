
from Taskc2 import Loading_and_processing
from Taskc4 import build_the_model, evaluate_and_report
from Taskc5 import multistep_and_multivariate
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX   
from sklearn.metrics import mean_absolute_error

# =================ARIMA model for multistep predictions.=======================
def arima_multistep_matrix_rolling(train_usd: np.ndarray,
                                   test_usd: np.ndarray,
                                   lookback: int,
                                   horizon: int,
                                   order=(5, 1, 0)) -> np.ndarray:
    # convert to 1d arrays
    train_usd = np.asarray(train_usd, dtype=float).reshape(-1)
    test_usd  = np.asarray(test_usd,  dtype=float).reshape(-1)

    # how many DL test windows exist? (aligns ARIMA with DL)
    N = len(test_usd) - lookback - horizon + 1
    
    #IF there are not enough data throw an error
    if N <= 0:
        raise ValueError("Not enough test data for given lookback + horizon.")
    
     # Allocate output matrix: one row per rolling position, `horizon` columns
    Y = np.zeros((N, horizon), dtype=float)

    # Initial history seen by the first ARIMA fit:
    # all training data followed by the first `lookback` points of test
    history = list(np.concatenate([train_usd, test_usd[:lookback]]))

     # Roll across test set
    for i in range(N):
        # Fit ARIMA to the current history
        model = ARIMA(history, order=order)
        fit   = model.fit(method_kwargs={"disp": True, "maxiter": 1000})

        # Make multistep forecast of length `horizon`
        fc    = fit.forecast(steps=horizon)

        # Store forecast row i
        Y[i, :] = np.asarray(fc, dtype=float)

        # Advance the rolling window by revealing the next true observation
        # so the next iteration uses a longer history
        history.append(test_usd[lookback + i])

    return Y


# =================ARIMA model for multistep predictions.=======================
def sarima_multistep_matrix_rolling(train_usd: np.ndarray,
                                    test_usd: np.ndarray,
                                    lookback: int,
                                    horizon: int,
                                    order=(1,1,1),
                                    seasonal_order=(0,1,1,5)) -> np.ndarray:
    # convert to 1d arrays
    train_usd = np.asarray(train_usd, dtype=float).reshape(-1)
    test_usd  = np.asarray(test_usd,  dtype=float).reshape(-1)
    # how many DL test windows exist? (aligns ARIMA with DL)
    N = len(test_usd) - lookback - horizon + 1

    #IF there are not enough data throw an error
    if N <= 0:
        raise ValueError("Not enough test data for given lookback + horizon.")
    
    # Allocate output matrix: one row per rolling position, `horizon` columns
    Y = np.zeros((N, horizon), dtype=float)

    # Initial history seen by the first ARIMA fit:
    # all training data followed by the first `lookback` points of test
    history = list(np.concatenate([train_usd, test_usd[:lookback]]))
    # Roll across test set
    for i in range(N):
        # Fit SARIMA to the current history
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit   = model.fit(method='lbfgs', maxiter=1000,disp=True)
        # Make multistep forecast of length `horizon`
        fc    = fit.forecast(steps=horizon)
        # Store forecast row i
        Y[i, :] = np.asarray(fc, dtype=float)
        # Advance the rolling window by revealing the next true observation
        # so the next iteration uses a longer history   
        history.append(test_usd[lookback + i])

    return Y



if __name__ == "__main__":
    # set inputs for the function.
    ticker = "CBA.AX"
    start  = "2015-01-01"
    end    = "2025-01-01"
    test_ratio = 0.2
    use_scale  = True
    feature_cols = ["open", "high", "low", "close", "adjclose", "volume"]
    # function which was imported from taskc2.py
    train_df, test_df, df, scalers = Loading_and_processing(
        ticker, start, end,
        split_method="ratio", test_size=test_ratio,
        scale=use_scale, feature_cols=feature_cols
    )

    print("Train/Test/Full:", len(train_df), len(test_df), len(df))

    kvalue = int(input("How many days would you prefer to predict? "))  # horizon
    lookback = 60

   
    cols_in = ["adjclose", "volume", "open", "close", "high", "low"]
    X_train, y_train = multistep_and_multivariate(train_df[cols_in], lookback=lookback, lookahead=kvalue)
    X_test,  y_test  = multistep_and_multivariate(test_df[cols_in],  lookback=lookback, lookahead=kvalue)
    print("Shapes  X_train/y_train:", X_train.shape, y_train.shape)
    print("Shapes  X_test /y_test :", X_test.shape,  y_test.shape)
    # Getting User inputs to build the model
    print("Choose a model:\n 1: LSTM \n 2: GRU \n 3: RNN")
    modeltype = int(input("Enter the model No: "))
    n_layers  = int(input("Enter No of Model Layers: "))
    units     = int(input("Enter Layer Size (units): "))
    epochs    = int(input("Enter No of epochs: "))
    batch     = int(input("Enter batch size: "))

    if modeltype == 1:
        Mtype, name = LSTM, "LSTM"
    elif modeltype == 2:
        Mtype, name = GRU, "GRU"
    else:
        Mtype, name = SimpleRNN, "RNN"
    
    model = build_the_model(Mtype, lookback=lookback,
                            n_layers=n_layers, units=units,
                            dropout=0.3, dense_layers=kvalue)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    # train 
    history = model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch,
        validation_data=(X_test, y_test),
        callbacks=[early_stop], verbose=1
    )
    #  predict on test sequences 
    y_pred = model.predict(X_test)
    N, K   = y_test.shape   # N test windows, K-step horizon


    #  invert scaling for target to get USD prices 
    # reload scalers to ensure same fit as preprocessing time
    scalers  = joblib.load("feature_scalers.pkl")
    t_scaler = scalers["adjclose"] # scaler used for the target

    # inverse-transform true and predicted targets to USD for reporting if needed
    y_test_inv = t_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(N, K)
    y_pred_inv = t_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(N, K)

    # also extract train and test adjclose in USD for ARIMA inputs
    train_usd = t_scaler.inverse_transform(train_df[["adjclose"]]).reshape(-1)
    test_usd  = t_scaler.inverse_transform(test_df[["adjclose"]]).reshape(-1)
    
    #classical baselines: ARIMA and SARIMA 
    arima_order = (5, 1, 0) 
    sarima_order = (1, 1, 1)
    sarima_seasonal = (0, 1, 1, 5) # period=5 (weekly pattern on trading days)  

    # rolling multi-step forecasts to align with DL evaluation windows
    Y_arima_usd = arima_multistep_matrix_rolling(train_usd, test_usd, lookback, kvalue, order=arima_order)
    # rescale ARIMA forecasts back to model scale for fair MAE comparison
    Y_arima_scaled = t_scaler.transform(Y_arima_usd.reshape(-1, 1)).reshape(Y_arima_usd.shape)


    Y_sarima_usd = sarima_multistep_matrix_rolling(train_usd, test_usd, lookback, kvalue, order=sarima_order, seasonal_order=sarima_seasonal)
    Y_sarima_scaled = t_scaler.transform(Y_sarima_usd.reshape(-1, 1)).reshape(Y_sarima_usd.shape)

    # alpha is the weight for DL prediction; 1 - alpha for classical model
    alpha = 0.5  # DL weight
    Y_ens_arima = alpha * y_pred + (1 - alpha) * Y_arima_scaled
    Y_ens_sarima = alpha * y_pred + (1 - alpha) * Y_sarima_scaled

    # per-horizon MAE on scaled space 
    print("\n==== MAE per horizon (Scaled 0–1) ====")
    for h in range(K):
        mae_a = mean_absolute_error(y_test[:, h], Y_arima_scaled[:, h])
        mae_s = mean_absolute_error(y_test[:, h], Y_sarima_scaled[:, h])
        mae_l = mean_absolute_error(y_test[:, h], y_pred[:, h])
        mae_e1 = mean_absolute_error(y_test[:, h], Y_ens_arima[:, h])
        mae_e2 = mean_absolute_error(y_test[:, h], Y_ens_sarima[:, h])
        print(f"t+{h+1}: ARIMA={mae_a:.6f}  SARIMA={mae_s:.6f}  {name}={mae_l:.6f}  ENS(ARIMA)={mae_e1:.6f}  ENS(SARIMA)={mae_e2:.6f}")

    # each figure compares Actual vs DL vs ARIMA/SARIMA and their ensembles
    for h in range(K):
        plt.figure()
        plt.plot(y_test[:, h], label=f"Actual t+{h+1}")
        plt.plot(y_pred[:, h], label=f"{name} t+{h+1}")
        plt.plot(Y_arima_scaled[:, h], label=f"ARIMA t+{h+1}")
        plt.plot(Y_ens_arima[:, h], label=f"Ensemble ARIMA t+{h+1}", linestyle="--")
        plt.legend()
        plt.title(f"t+{h+1} Forecast: Actual vs Models (ARIMA + {name})")
        plt.xlabel("Test samples"); plt.ylabel("Scaled price")
        plt.tight_layout()
        plt.savefig(f"Taskc6-plots/ARIMA t+{h+1} + {name} plot.png"); plt.show()

        plt.figure()
        plt.plot(y_test[:, h], label=f"Actual t+{h+1}")
        plt.plot(y_pred[:, h], label=f"{name} t+{h+1}")
        plt.plot(Y_sarima_scaled[:, h], label=f"SARIMA t+{h+1}")
        plt.plot(Y_ens_sarima[:, h], label=f"Ensemble SARIMA t+{h+1}", linestyle="--")
        plt.legend()
        plt.title(f"t+{h+1} Forecast: Actual vs Models (SARIMA + {name})")
        plt.xlabel("Test samples"); plt.ylabel("Scaled price")
        plt.tight_layout()
        plt.savefig(f"Taskc6-plots/SARIMA t+{h+1} + {name} plot.png"); plt.show()

    print("\n✅ ARIMA/SARIMA ensemble completed successfully.")