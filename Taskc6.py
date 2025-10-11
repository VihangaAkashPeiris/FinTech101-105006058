# ===== TaskC6.py (Ensemble: DL + ARIMA + SARIMA) =====
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
from statsmodels.tsa.statespace.sarimax import SARIMAX   # ✅ for SARIMA
from sklearn.metrics import mean_absolute_error

# ---------------- ARIMA helper: rolling multistep, aligned to DL windows ----------------
def arima_multistep_matrix_rolling(train_usd: np.ndarray,
                                   test_usd: np.ndarray,
                                   lookback: int,
                                   horizon: int,
                                   order=(5, 1, 0)) -> np.ndarray:
    """
    Returns Y_arima_usd with shape (N, horizon) aligned to DL windows:
      N = len(test_usd) - lookback - horizon + 1
      Row i forecasts t+1..t+horizon at the forecast origin of DL sample i.
    """
    train_usd = np.asarray(train_usd, dtype=float).reshape(-1)
    test_usd  = np.asarray(test_usd,  dtype=float).reshape(-1)
    N = len(test_usd) - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Not enough test data for given lookback + horizon.")

    Y = np.zeros((N, horizon), dtype=float)
    history = list(np.concatenate([train_usd, test_usd[:lookback]]))

    for i in range(N):
        model = ARIMA(history, order=order)
        fit   = model.fit(method_kwargs={"disp": True, "maxiter": 1000})
        fc    = fit.forecast(steps=horizon)
        Y[i, :] = np.asarray(fc, dtype=float)
        history.append(test_usd[lookback + i])

    return Y


# ==== NEW SARIMA FUNCTION =====================================================
def sarima_multistep_matrix_rolling(train_usd: np.ndarray,
                                    test_usd: np.ndarray,
                                    lookback: int,
                                    horizon: int,
                                    order=(2,1,2),
                                    seasonal_order=(1,1,1,5)) -> np.ndarray:
    """
    Rolling SARIMA: produces forecasts aligned to DL windows (same as ARIMA).
    Useful when data has seasonal patterns (e.g., weekly cycles).

    order: (p,d,q)
    seasonal_order: (P,D,Q,s) -> s=5 means weekly seasonality (5 trading days)
    """
    train_usd = np.asarray(train_usd, dtype=float).reshape(-1)
    test_usd  = np.asarray(test_usd,  dtype=float).reshape(-1)
    N = len(test_usd) - lookback - horizon + 1
    if N <= 0:
        raise ValueError("Not enough test data for given lookback + horizon.")

    Y = np.zeros((N, horizon), dtype=float)
    history = list(np.concatenate([train_usd, test_usd[:lookback]]))

    for i in range(N):
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit   = model.fit(method='lbfgs', maxiter=1000,disp=True)
        fc    = fit.forecast(steps=horizon)
        Y[i, :] = np.asarray(fc, dtype=float)
        history.append(test_usd[lookback + i])

    return Y
# ==============================================================================


if __name__ == "__main__":
    # =================== 1) Data ===================
    ticker = "CBA.AX"
    start  = "2015-01-01"
    end    = "2025-01-01"
    test_ratio = 0.2
    use_scale  = True
    feature_cols = ["open", "high", "low", "close", "adjclose", "volume"]

    train_df, test_df, df, scalers = Loading_and_processing(
        ticker, start, end,
        split_method="ratio", test_size=test_ratio,
        scale=use_scale, feature_cols=feature_cols
    )

    print("Train/Test/Full:", len(train_df), len(test_df), len(df))

    kvalue = int(input("How many days would you prefer to predict? "))  # horizon
    lookback = 60

    # =================== 2) Sequences (multistep + multivariate) ===================
    cols_in = ["adjclose", "volume", "open", "close", "high", "low"]
    X_train, y_train = multistep_and_multivariate(train_df[cols_in], lookback=lookback, lookahead=kvalue)
    X_test,  y_test  = multistep_and_multivariate(test_df[cols_in],  lookback=lookback, lookahead=kvalue)
    print("Shapes  X_train/y_train:", X_train.shape, y_train.shape)
    print("Shapes  X_test /y_test :", X_test.shape,  y_test.shape)

    # =================== 3) Build & Train DL model ===================
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
    history = model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch,
        validation_data=(X_test, y_test),
        callbacks=[early_stop], verbose=1
    )
    y_pred = model.predict(X_test)
    N, K   = y_test.shape

    # =================== 4) Scalers / inverse ===================
    scalers  = joblib.load("feature_scalers.pkl")
    t_scaler = scalers["adjclose"]
    y_test_inv = t_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(N, K)
    y_pred_inv = t_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(N, K)

    # =================== 5) ARIMA & SARIMA rolling multistep ===================
    train_usd = t_scaler.inverse_transform(train_df[["adjclose"]]).reshape(-1)
    test_usd  = t_scaler.inverse_transform(test_df[["adjclose"]]).reshape(-1)

    arima_order = (5, 1, 0)
    sarima_order = (1, 1, 1)
    sarima_seasonal = (0, 1, 1, 5)

    # ARIMA
    Y_arima_usd = arima_multistep_matrix_rolling(train_usd, test_usd, lookback, kvalue, order=arima_order)
    Y_arima_scaled = t_scaler.transform(Y_arima_usd.reshape(-1, 1)).reshape(Y_arima_usd.shape)

    # SARIMA
    Y_sarima_usd = sarima_multistep_matrix_rolling(train_usd, test_usd, lookback, kvalue, order=sarima_order, seasonal_order=sarima_seasonal)
    Y_sarima_scaled = t_scaler.transform(Y_sarima_usd.reshape(-1, 1)).reshape(Y_sarima_usd.shape)

    # =================== 6) Ensemble per horizon ===================
    alpha = 0.5  # DL weight
    Y_ens_arima = alpha * y_pred + (1 - alpha) * Y_arima_scaled
    Y_ens_sarima = alpha * y_pred + (1 - alpha) * Y_sarima_scaled

    # =================== 7) Metrics ===================
    print("\n==== MAE per horizon (Scaled 0–1) ====")
    for h in range(K):
        mae_a = mean_absolute_error(y_test[:, h], Y_arima_scaled[:, h])
        mae_s = mean_absolute_error(y_test[:, h], Y_sarima_scaled[:, h])
        mae_l = mean_absolute_error(y_test[:, h], y_pred[:, h])
        mae_e1 = mean_absolute_error(y_test[:, h], Y_ens_arima[:, h])
        mae_e2 = mean_absolute_error(y_test[:, h], Y_ens_sarima[:, h])
        print(f"t+{h+1}: ARIMA={mae_a:.6f}  SARIMA={mae_s:.6f}  {name}={mae_l:.6f}  ENS(ARIMA)={mae_e1:.6f}  ENS(SARIMA)={mae_e2:.6f}")

    # =================== 8) Plots ===================
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
        plt.savefig(f"Taskc6-plots/ARIMA t+{h+1} plot.png"); plt.show()

        plt.figure()
        plt.plot(y_test[:, h], label=f"Actual t+{h+1}")
        plt.plot(y_pred[:, h], label=f"{name} t+{h+1}")
        plt.plot(Y_sarima_scaled[:, h], label=f"SARIMA t+{h+1}")
        plt.plot(Y_ens_sarima[:, h], label=f"Ensemble SARIMA t+{h+1}", linestyle="--")
        plt.legend()
        plt.title(f"t+{h+1} Forecast: Actual vs Models (SARIMA + {name})")
        plt.xlabel("Test samples"); plt.ylabel("Scaled price")
        plt.tight_layout()
        plt.savefig(f"Taskc6-plots/SARIMA t+{h+1} plot.png"); plt.show()

    print("\n✅ ARIMA/SARIMA ensemble completed successfully.")