# File: Taskc2.py
# Author: Vihanga Akash Peiris
#Student ID: 105006058

#need to download the following libraries to sucessfully run my code:
#pip install pandas
#pip install numpy
#pip install yfinance
#pip install scikit-learn

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

# This is the function  " load_and_processing())"that fullfill all the requirements in the task C2.
def Loading_and_processing(
    ticker: str, #. This is the ticker with the type hint String ("CBA.AX")
    start: str, #. This is the start parameter for start date withh the type hint string.
    end: str,#. This is the end parameter for end date withh the type hint string.
    *, #. The begining of key words only arguments.
    interval: str = "1d", #Time period of the data per day (1d), per month (1m)or per year (1y).
    auto_adjust: bool = False, # To avoid the auto adjusting because we need the adj close column
    split_method: str = "ratio",   # this is the split method and can be one of the follow  ings:"ratio" | "date" | "random"
    test_size: float = 0.2,        # split ratio for test data under split_mothod="ratio".
    split_date: str | None = None, # split date for split_method="date"
    low: float = 0.15,             # the lowest value that can be generated randomly for  split_method="random"
    high: float = 0.30,            # the highest value that can be generated randomly for  split_method="random"
    random_state: int = 314,       # Used this just to keep a fixed random  value.
    scale: bool = False,                    # used for enable/disable scaling
    feature_cols: list[str] | None = None,  # feature column list :which columns to scale
):
    #  If df_raw data frame is missing or empty it returns a empty dataframe witout crashing and this will make a copy of the dataframe so the original dataframe is safe.
    def _normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            return pd.DataFrame()
        df = df_raw.copy()

        # This will checks for the multiIndexes in both levels (first and last) and slice the CBA.AX.
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(key=ticker, axis=1, level=-1) # check in the last level in the columns
            except (KeyError, ValueError):
                try:
                    df = df.xs(key=ticker, axis=1, level=0) # check in the fist level in the columns
                except (KeyError, ValueError):
                    df.columns = [str(c[0]) for c in df.columns]

        # This will standardize the column names as follows:
        df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adjclose", "Volume": "volume",
        }, inplace=True)

        # This checks for the time zone indexes in df.index and remove it if available and sort it according to the dates.
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)

        # This ensures adj close exists if not it will create a one and get close column's data.
        if "adjclose" not in df.columns:
            if auto_adjust and "close" in df.columns:
                df["adjclose"] = df["close"]
            else:
                df["adjclose"] = df.get("close", pd.NA)

        # This will handle the NaNs and drop rows with NaNs in key cols and also this converts data to numeric.
        key_cols = [c for c in ("open", "high", "low", "close", "adjclose", "volume") if c in df.columns]
        for c in key_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=key_cols, inplace=True)

        return df
    # This will download the stock data according to the start and end date range
    def _download_yf(s: str, e: str) -> pd.DataFrame:
        raw = yf.download(
            ticker, start=s, end=e,
            interval=interval, auto_adjust=auto_adjust, progress=False
        )
        
        return _normalize(raw)

    # ----------This part will store the data locally and load when neccessary--------
    
    cache_dir = Path("Stock_CSV") # Define the folder name that stores stock data
    cache_dir.mkdir(parents=True, exist_ok=True) # This will Create the directory if it doesn’t already exist
    safe_ticker = ticker.replace("/", "-").replace(".", "_") # File names cant have ".,-" kind of symbols so this will handle it.(CBA.AX->CBA_AX)
    cache_path = cache_dir / f"{safe_ticker}_{interval}_adj{int(bool(auto_adjust))}.csv" # This define the saved file name using ticker, interval, and auto adjust etc.

    
    req_start = pd.to_datetime(start) # assign the input start date for req_start
    req_end   = pd.to_datetime(end) # assign the input end date for req_end
    # this checks whether the request data is available in the saved file or not 
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        cached = _normalize(cached) # clean the saved dataframe
    else:
        cached = pd.DataFrame()

    # Determine i`f we need to fetch extra data
    # This will check whether the dataframe is empty or the request date is not in the dataframe and it should locate before the first index in the data frame.
    need_left  = cached.empty or req_start < cached.index.min()
    # This will check whether the dataframe is empty or the request date is not in the dataframe and it should locate after  the last index in the data frame.
    need_right = cached.empty or req_end   > cached.index.max()

    pieces = [] # This will collect DataFrames (cache + newly downloaded pieces).
    # If cache already has some data, include it first.
    if not cached.empty:
        pieces.append(cached)
        
    # If we need data earlier than cached range it will append those data.
    if need_left:
        left_end = (cached.index.min() - timedelta(days=1)).strftime("%Y-%m-%d") if not cached.empty else end
        pieces.append(_download_yf(start, left_end))
    # If we need data later than cached range it will append those data.
    if need_right:
        right_start = (cached.index.max() + timedelta(days=1)).strftime("%Y-%m-%d") if not cached.empty else start
        pieces.append(_download_yf(right_start, end))
    # if we have at least one piece (cache or new data) this will Concatenate everything into a single DataFrame
    if pieces:
        merged = pd.concat(pieces)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()#Drop duplicate dates and sort everything by date
        merged.to_csv(cache_path, index_label="Date")
    else:
        merged = cached

    # slice exactly the requested window for downstream split and df contains only the requested time window
    df = merged.loc[(merged.index >= req_start) & (merged.index <= req_end)].copy()
    if df.empty:
        raise ValueError(f"No data available for {ticker} between {start} and {end}.")

    # --------------- splitting ---------------
    n = len(df)
    if n < 2: # Must have atleast 2 rows to split this checks that.
        raise ValueError("Not enough rows to split.")
    # Split the data using ratio if the input ration stays between 0.0 and 0.9
    if split_method == "ratio": 
        if not (0.0 < test_size < 0.9):
            raise ValueError("test_size must be between 0 and 0.9.")
        split_idx = int((1.0 - test_size) * n)
        if split_idx <= 0 or split_idx >= n:
            raise ValueError("Split index produced empty train or test set; adjust test_size.")
        train_df = df.iloc[:split_idx].copy() # splitted train dataframe
        test_df  = df.iloc[split_idx:].copy() # splitted test dataframe
    # Split the dataframe using the method "date".
    elif split_method == "date":
        if not split_date:
            raise ValueError("split_date must be provided for split_method='date'.")
        cutoff = pd.to_datetime(split_date)
        train_df = df[df.index <  cutoff].copy() # splitted train dataframe
        test_df  = df[df.index >= cutoff].copy() # splitted test dataframe
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Split date produced empty train or test set; choose a different date.")
    # split the dataframe randomly
    elif split_method == "random":
        rng = np.random.default_rng(random_state)
        tsize = float(rng.uniform(low, high))
        mask = rng.random(n) >= tsize   # True->train, False->test
        train_df = df[mask].sort_index().copy() # splitted train dataframe
        test_df  = df[~mask].sort_index().copy() # splitted test dataframe
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Random split produced an empty set. Adjust low/high or data size.")
    else:
        raise ValueError("split_method must be 'ratio', 'date', or 'random'.")

    # --------------- scaling ---------------
    # This part will do the scaling of the feature columns usig the minmaxscaler method.
    scalers: dict[str, MinMaxScaler] = {} # dictionary that maps column names to MinMaxScaler objects.
    if scale:
        # default to common OHLCV columns that exist
        if feature_cols is None:
            feature_cols = [c for c in ("open","high","low","close","adjclose","volume") if c in df.columns]

        for col in feature_cols:
            if col not in train_df.columns:
                continue
            scaler = MinMaxScaler()  # 0..1
            # fit on TRAIN ONLY
            train_df[col] = scaler.fit_transform(train_df[[col]])
            # transform TEST with same scaler
            if col in test_df.columns and len(test_df) > 0:
                test_df[col] = scaler.transform(test_df[[col]])
            scalers[col] = scaler
            
        joblib.dump(scalers, "feature_scalers.pkl")
    else:
        scalers = {}

    return train_df, test_df, df, scalers

if __name__ == "__main__": 
# this will ensure user enters the date in a  correct format
    def _date_validation(s: str) -> bool:
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return True
        except Exception:
            return False




# Promt the user to enter the start and end dates.
    start = input("Please enter Start date (yyyy-mm-dd): ")
    if not _date_validation(start):
        raise ValueError("Start date must be in yyyy-mm-dd format")
    end = input("Please enter the end date (yyyy-mm-dd): ")
    if not _date_validation(end):
        raise ValueError("End date must be in yyyy-mm-dd format")

# This will ask the  user whether they want to scale feature collumns or not
    use_scale = input("Scale features to 0–1? (y/N): ").strip().lower() == "y"
# leave blank to auto-pick common OHLCV columns; or comma-separated list (e.g., adjclose,volume)
    cols_in = input("Columns to scale (blank=auto; e.g. adjclose,open,high,low,close,volume): ").strip()
    feature_cols = [c.strip() for c in cols_in.split(",")] if cols_in else None
# This will ask the user to choose a data splitting method
    print("Choose split method:\n1: By Ratio\n2: By Date\n3: By Random")
    option = int(input("Select from the above options: "))
# Common wealth Bank stocks
    ticker = "CBA.AX"
# If user choose the split method "ratio" this will ask the ration between 0.0-0.9 to split the data.
    if option == 1:
        test_ratio = float(input("Please enter the test ratio (e.g., 0.2 for 20%): "))
        train_df, test_df, df, scalers = Loading_and_processing(
        ticker, start, end,
        split_method="ratio", test_size=test_ratio,
        scale=use_scale, feature_cols=feature_cols
    )
# if user choose the split method - "date" it will ask the users to enter a prefer starting date for  test data
    elif option == 2:
        split_date = input("Please enter the starting date of test (yyyy-mm-dd): ")
        train_df, test_df, df, scalers = Loading_and_processing(
            ticker, start, end,
            split_method="date", split_date=split_date,
            scale=use_scale, feature_cols=feature_cols
    )
# If user choose the split method = "random" it will randomly split the dataframe
    elif option == 3:
        train_df, test_df, df, scalers = Loading_and_processing(
            ticker, start, end,
            split_method="random", low=0.15, high=0.30, random_state=314,
            scale=use_scale, feature_cols=feature_cols
    )
    else:
        raise ValueError("Invalid option. Choose 1, 2, or 3.")

# these print statements were used to check the accuracy of the splitted methods
    print(f"No of train rows: {len(train_df)}")
    print(f"No of test  rows: {len(test_df)}")
    print("Train period:", train_df.index.min().date(), "→", train_df.index.max().date())
    print("Test  period:",  test_df.index.min().date(),  "→", test_df.index.max().date())
# this will print the first and last five lined of the dataframe
    print("\nTrain head:\n", train_df.head())
    print("\nTest head:\n", test_df.head())

# If scaling was enabled, this wi;; show which columns were scaled
    if use_scale:
        print("\nScalers available for columns:", list(scalers.keys()))
pass