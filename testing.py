# === Task C4: Multivariate LSTM using all features to predict next adjclose ===
from Taskc2 import Loading_and_processing

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


# ---------- build sliding windows from MULTIPLE features ----------
def make_sequences_multi(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    horizon: int = 1,
):
    """
    X shape -> (num_samples, lookback, n_features)
    y shape -> (num_samples,)
    """
    feat = df[feature_cols].values.astype(float)  # (N, n_features)
    tgt  = df[target_col].values.astype(float)    # (N,)

    n = len(df)
    n_features = len(feature_cols)
    num_samples = n - lookback - horizon + 1
    if num_samples <= 0:
        raise ValueError(
            f"Not enough rows: need > lookback+horizon-1, got {n}"
        )

    X = np.zeros((num_samples, lookback, n_features), dtype=float)
    y = np.zeros((num_samples,), dtype=float)

    for i in range(num_samples):
        X[i] = feat[i : i + lookback, :]                      # past lookback rows (all features)
        y[i] = tgt[i + lookback + horizon - 1]                # next target value

    return X, y


# ---------- model builder (LSTM) ----------
def build_the_model(
    lookback: int,
    n_features: int,
    n_layers: int = 2,
    units: int = 50,
    dropout: float = 0.3,
) -> tf.keras.Model:
    model = Sequential()
    model.add(
        LSTM(units, return_sequences=(n_layers > 1),
             input_shape=(lookback, n_features))
    )
    model.add(Dropout(dropout))

    # middle stacked LSTM layers (if any)
    for _ in range(1, n_layers - 1):
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))

    # last LSTM layer
    if n_layers > 1:
        model.add(LSTM(units))
        model.add(Dropout(dropout))

    model.add(Dense(1))  # predict next adjclose
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


if __name__ == "__main__":
    # ----------- config -----------
    ticker = "CBA.AX"
    start = "2015-01-01"
    end   = "2021-01-01"

    test_ratio = 0.2
    use_scale = True  # relies on Taskc2 scaling
    feature_cols = ["open", "high", "low", "close", "adjclose", "volume"]
    target_col   = "adjclose"

    lookback = 60
    horizon  = 1

    # ----------- load & split (Task C2) -----------
    train_df, test_df, df, scalers = Loading_and_processing(
        ticker, start, end,
        split_method="ratio", test_size=test_ratio,
        scale=use_scale, feature_cols=feature_cols
    )

    # ----------- sequences -----------
    X_train, y_train = make_sequences_multi(train_df, feature_cols, target_col, lookback, horizon)
    X_test,  y_test  = make_sequences_multi(test_df,  feature_cols, target_col, lookback, horizon)

    print("Train:", X_train.shape, y_train.shape)  # e.g. (Ntrain, 60, 6) (Ntrain,)
    print("Test :", X_test.shape,  y_test.shape)   # e.g. (Ntest,  60, 6) (Ntest,)

    # ----------- model -----------
    n_features = len(feature_cols)
    model = build_the_model(
        lookback=lookback,
        n_features=n_features,
        n_layers=2,
        units=64,
        dropout=0.3
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=250,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # ----------- predict -----------
    y_pred = model.predict(X_test).ravel()

    # ----------- optional: inverse-scale to real prices -----------
    # Taskc2 should return a dict of scalers per column; try to invert if available.
    def inverse_if_possible(arr_1d: np.ndarray, col: str) -> np.ndarray:
        try:
            sc = scalers.get(col, None)
            if sc is None:
                return arr_1d
            return sc.inverse_transform(arr_1d.reshape(-1, 1)).ravel()
        except Exception:
            # fall back to scaled values if inverse not possible
            return arr_1d

    y_test_real = inverse_if_possible(y_test, target_col)
    y_pred_real = inverse_if_possible(y_pred, target_col)

    # ----------- simple metrics -----------
    rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
    mae  = float(np.mean(np.abs(y_pred - y_test)))
    print(f"Scaled RMSE: {rmse:.6f} | Scaled MAE: {mae:.6f}")

    # If inverse-scaling succeeded, print real-units metrics too
    if not np.allclose(y_test_real, y_test):
        rmse_real = float(np.sqrt(np.mean((y_pred_real - y_test_real)**2)))
        mae_real  = float(np.mean(np.abs(y_pred_real - y_test_real)))
        print(f"Real-price RMSE: {rmse_real:.4f} | Real-price MAE: {mae_real:.4f}")

    # ----------- plot -----------
    plt.figure(figsize=(10, 5))
    if not np.allclose(y_test_real, y_test):
        plt.plot(y_test_real, label="Actual (price)")
        plt.plot(y_pred_real, label="Predicted (price)")
        plt.ylabel("Price")
    else:
        plt.plot(y_test, label="Actual (scaled)")
        plt.plot(y_pred, label="Predicted (scaled)")
        plt.ylabel("Scaled price")

    plt.xlabel("Time (test samples)")
    plt.title("Actual vs Predicted (Test) — Multivariate LSTM")
    plt.legend()
    plt.tight_layout()
    plt.show()