import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def load_data_yf(ticker: str ,start: str ,end: str, interval: str = "1d" , auto_adjust: bool = False) ->pd.DataFrame:
    df = yf.download (ticker,start = start, end = end , interval = interval, auto_adjust=auto_adjust, progress = False)
    
    if df is None or df.empty:
        raise ValueError(f"no data returned for {ticker} between {start} and {end}.")
    # If yfinance returned MultiIndex columns, keep only this ticker (or flatten)
    if isinstance(df.columns, pd.MultiIndex):
    # Preferred: select the columns for THIS ticker level
    # yfinance usually uses level 0 = field, level 1 = ticker
    # but some versions swap; try both safely.
        try:
        # try last level as ticker
            df = df.xs(key=ticker, axis=1, level=-1)
        except (KeyError, ValueError):
            try:
            # try first level as ticker
                df = df.xs(key=ticker, axis=1, level=0)
            except (KeyError, ValueError):
            # fallback: just take the first element of each tuple ('Open','CBA.AX') -> 'Open'
                df.columns = [str(col[0]) for col in df.columns]

    
    colmap = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjclose",
    "Volume": "volume",
    }
    df.rename(columns=colmap, inplace=True)

    if "adjclose" not in df.columns:
        if auto_adjust and "close" in df.columns:
            df["adjclose"] = df["close"]
        else:
            df["adjclose"] = df.get("close", pd.NA)
    
    df = df.sort_index()
    # make a plain 'date' column (helpful later for splits/merges)
    if getattr(df.index, "tz", None) is not None:
        df["date"] = df.index.tz_localize(None)
    else:
        df["date"] = df.index

    # coerce numerics (avoids stray strings down the line)
    for c in ("open", "high", "low", "close", "adjclose", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # handle NANs by dropping rows
    df.dropna(subset=["open", "high", "low", "close", "adjclose", "volume"], inplace=True)


    return df


#############################################
def date_format_check (date_text):
    try:
        datetime.strptime(date_text, "%Y-%m-%d") 
        return True
    except:
        return False
    


start = str(input ("Please enter Start date (yyyy-mm-dd) :"))
if not (date_format_check(start)):
    raise ValueError("Start date must be in yyyy-mm-dd format")

end = str(input("Please enter the end date (yyyy-mm-dd):"))
if not (date_format_check(end) ):
    raise ValueError("End date must be in yyyy-mm-dd format")

#############################################

def split_by_ratio(df: pd.DataFrame, test_size:float ):
    n = len(df)
    if n < 2:
        raise ValueError("Not enough rows to split.")

    if not (0.0 < test_size < 0.9):
        raise ValueError("test_size must be between 0 and 0.9.")

    split_idx = int((1.0 - test_size) * n)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Split index produced empty train or test set; adjust test_size.")

    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    return train_df, test_df
#############################################################
def split_by_date(df: pd.DataFrame, split_date : str):
    cutoff = pd.to_datetime(split_date)

    train_df = df[df.index<cutoff]
    test_df = df[df.index>=cutoff]

    return train_df,test_df


##############################################################
def split_by_random(df:pd.DataFrame, low =0.15, high =0.30, random_state=314):
    rng = np.random.default_rng(random_state)
    test_size = float(rng.uniform(low, high)) 
# Randomly assign rows to train/test using the chosen ratio
    mask = rng.random(len(df)) >= test_size       # True -> train, False -> test
    train_df = df[mask].sort_index().copy()
    test_df  = df[~mask].sort_index().copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Split produced an empty set. Adjust low/high or check data size.")

    return train_df, test_df



df = load_data_yf("CBA.AX" , start = start, end = end)

print ("what method would you like to choose to split the data into train/test?\n1: By Ratio\n2: By Date\n3: By Random")
option=int(input("Select from the above options:"))

if option == 1:
    test_size=float(input("Please Enter the test ratio (0.2)->20% : "))
    train_df, test_df=split_by_ratio(df,test_size)
    print(f"No of train data :{ len(train_df)}")
    print(f"No of train data :{ len(test_df)}")
    print("Train period:", train_df.index.min().date(), "→", train_df.index.max().date())
    print("Test period:", test_df.index.min().date(), "→", test_df.index.max().date())
if option == 2:
    split_date = str(input("Please enter the starting date of test (yyyy-mm-dd) :"))
    train_df,test_df = split_by_date(df,split_date)
    print(f"No of train data :{ len(train_df)}")
    print(f"No of train data :{ len(test_df)}")
    print("Train period:", train_df.index.min().date(), "→", train_df.index.max().date())
    print("Test period:", test_df.index.min().date(), "→", test_df.index.max().date())
if option == 3:
    train_df,test_df = split_by_random(df,low =0.15, high =0.30, random_state=314)
    print(f"No of train data :{ len(train_df)}")
    print(f"No of train data :{ len(test_df)}")
    print("Train period:", train_df.index.min().date(), "→", train_df.index.max().date())
    print("Test period:", test_df.index.min().date(), "→", test_df.index.max().date())


print(df.head())
print(df.tail())