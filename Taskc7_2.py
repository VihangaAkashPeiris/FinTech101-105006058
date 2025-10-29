import numpy as np
import pandas as pd


# Load the merged prices + daily sentiment file with only the needed columns
DF = pd.read_csv(
    "Taskc7/final_prices_&_sentiment.csv",
    usecols=["date", "adjclose", "vader_mean", "finbert_mean"]
)

# Make sure date is a datetime, sort chronologically, and tidy the index
DF["date"] = pd.to_datetime(DF["date"])
DF = DF.sort_values("date").reset_index(drop=True)
print(" Data loaded and sorted")
print(DF.head(3))


# Core price features
DF["ret_1d"] = DF["adjclose"].pct_change(1)                           # daily return
DF["ma_5"] = DF["adjclose"].rolling(window=5).mean()                  # 5-day MA
DF["ma_10"] = DF["adjclose"].rolling(window=10).mean()                # 10-day MA
DF["mom_5"] = DF["adjclose"] / DF["adjclose"].shift(5) - 1            # 5-day momentum
DF["vol_5"] = DF["ret_1d"].rolling(window=5).std()                    # 5-day volatility

# RSI (14-day) using simple rolling averages
window = 14
delta = DF["adjclose"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=window).mean()
avg_loss = pd.Series(loss).rolling(window=window).mean()
RS = avg_gain / avg_loss
DF["RSI_14"] = 100 - (100 / (1 + RS))

print(" Price-based indicators computed")


# Sentiment deltas and 1-day lags for prediction features
DF["vader_change"]   = DF["vader_mean"].diff()
DF["finbert_change"] = DF["finbert_mean"].diff()

# Shift sentiment to align yesterday’s signal with today’s outcome
DF["vader_mean_shifted"]     = DF["vader_mean"].shift(1)
DF["finbert_mean_shifted"]   = DF["finbert_mean"].shift(1)
DF["vader_change_shifted"]   = DF["vader_change"].shift(1)
DF["finbert_change_shifted"] = DF["finbert_change"].shift(1)

print(" Sentiment change and shifted features added")


# Define supervised targets
DF["ret_next_1d"] = DF["adjclose"].shift(-1) / DF["adjclose"] - 1     # next-day return

# Classification target: 1 if tomorrow is higher than today, else 0
DF["target"] = (DF["adjclose"].shift(-1) > DF["adjclose"]).astype(int)

# Remove rows made incomplete by rolling, diff, and shift
DF = DF.dropna().reset_index(drop=True)


# Basic class balance check
n_total = len(DF)
n_up = int(DF["target"].sum())
n_down = n_total - n_up

print("\n Target column created successfully!")
print(f"Total samples: {n_total}")
print(f"Up days (target=1): {n_up}  |  Down days (target=0): {n_down}")
print(f"Up ratio: {n_up / n_total:.2%}\n")

# Quick peek at targets and a numeric summary for sanity
print("Sample preview:")
print(DF[["date", "adjclose", "ret_next_1d", "target"]].head(10))
print("\nFeature summary (numeric columns):")
print(DF.describe().T)


# Persist the feature set for modelling
output_path = "Taskc7/DF_with_features_and_target.csv"
DF.to_csv(output_path, index=False)
print(f"\n Saved -> {output_path}")
print(" All features + target ready for modelling!")