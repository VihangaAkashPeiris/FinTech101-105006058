import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
#load
def load_data_yf(
    ticker: str,
    start: str,
    end: str,
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
    split_method: str = "ratio",   # "ratio" | "date" | "random"
    test_size: float = 0.2,        # used if split_method="ratio"
    split_date: str | None = None, # used if split_method="date"
    low: float = 0.15,             # used if split_method="random"
    high: float = 0.30,
    random_state: int = 314,
    scale: bool = False,                    # NEW: enable/disable scaling
    feature_cols: list[str] | None = None,  # NEW: which columns to scale
):
    # ---------------- helpers ----------------
    def _normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            return pd.DataFrame()
        df = df_raw.copy()

        # flatten MultiIndex from yfinance if present
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(key=ticker, axis=1, level=-1)
            except (KeyError, ValueError):
                try:
                    df = df.xs(key=ticker, axis=1, level=0)
                except (KeyError, ValueError):
                    df.columns = [str(c[0]) for c in df.columns]

        # standardize column names
        df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adjclose", "Volume": "volume",
        }, inplace=True)

        # tz-naive, sorted index
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)

        # ensure adjclose
        if "adjclose" not in df.columns:
            if auto_adjust and "close" in df.columns:
                df["adjclose"] = df["close"]
            else:
                df["adjclose"] = df.get("close", pd.NA)

        # numerics + drop rows with NaNs in key cols
        key_cols = [c for c in ("open", "high", "low", "close", "adjclose", "volume") if c in df.columns]
        for c in key_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=key_cols, inplace=True)

        return df

    def _download_yf(s: str, e: str) -> pd.DataFrame:
        raw = yf.download(
            ticker, start=s, end=e,
            interval=interval, auto_adjust=auto_adjust, progress=False
        )
        return _normalize(raw)

    # --------------- cache setup ---------------
    cache_dir = Path("Stock_CSV")
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("/", "-").replace(".", "_")
    cache_path = cache_dir / f"{safe_ticker}_{interval}_adj{int(bool(auto_adjust))}.csv"

    # --------------- load/extend cache ---------------
    req_start = pd.to_datetime(start)
    req_end   = pd.to_datetime(end)

    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        cached = _normalize(cached)
    else:
        cached = pd.DataFrame()

    # Determine if we need to fetch extra data
    need_left  = cached.empty or req_start < cached.index.min()
    need_right = cached.empty or req_end   > cached.index.max()

    pieces = []
    if not cached.empty:
        pieces.append(cached)

    if need_left:
        left_end = (cached.index.min() - timedelta(days=1)).strftime("%Y-%m-%d") if not cached.empty else end
        pieces.append(_download_yf(start, left_end))

    if need_right:
        right_start = (cached.index.max() + timedelta(days=1)).strftime("%Y-%m-%d") if not cached.empty else start
        pieces.append(_download_yf(right_start, end))

    if pieces:
        merged = pd.concat(pieces)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_csv(cache_path, index_label="Date")
    else:
        merged = cached

    # slice exactly the requested window for downstream split
    df = merged.loc[(merged.index >= req_start) & (merged.index <= req_end)].copy()
    if df.empty:
        raise ValueError(f"No data available for {ticker} between {start} and {end}.")

    # --------------- splitting ---------------
    n = len(df)
    if n < 2:
        raise ValueError("Not enough rows to split.")

    if split_method == "ratio":
        if not (0.0 < test_size < 0.9):
            raise ValueError("test_size must be between 0 and 0.9.")
        split_idx = int((1.0 - test_size) * n)
        if split_idx <= 0 or split_idx >= n:
            raise ValueError("Split index produced empty train or test set; adjust test_size.")
        train_df = df.iloc[:split_idx].copy()
        test_df  = df.iloc[split_idx:].copy()

    elif split_method == "date":
        if not split_date:
            raise ValueError("split_date must be provided for split_method='date'.")
        cutoff = pd.to_datetime(split_date)
        train_df = df[df.index <  cutoff].copy()
        test_df  = df[df.index >= cutoff].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Split date produced empty train or test set; choose a different date.")

    elif split_method == "random":
        rng = np.random.default_rng(random_state)
        tsize = float(rng.uniform(low, high))
        mask = rng.random(n) >= tsize   # True->train, False->test
        train_df = df[mask].sort_index().copy()
        test_df  = df[~mask].sort_index().copy()
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Random split produced an empty set. Adjust low/high or data size.")
    else:
        raise ValueError("split_method must be 'ratio', 'date', or 'random'.")

    # --------------- optional scaling ---------------
    scalers: dict[str, MinMaxScaler] = {}
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
    else:
        scalers = {}

    return train_df, test_df, df, scalers


 # --- validate date format (yyyy-mm-dd) ---
def _date_validation(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except Exception:
        return False




# --- get dates ---
start = input("Please enter Start date (yyyy-mm-dd): ")
if not _date_validation(start):
    raise ValueError("Start date must be in yyyy-mm-dd format")
end = input("Please enter the end date (yyyy-mm-dd): ")
if not _date_validation(end):
    raise ValueError("End date must be in yyyy-mm-dd format")

# --- scaling options (optional) ---
use_scale = input("Scale features to 0–1? (y/N): ").strip().lower() == "y"
# leave blank to auto-pick common OHLCV columns; or comma-separated list (e.g., adjclose,volume)
cols_in = input("Columns to scale (blank=auto; e.g. adjclose,open,high,low,close,volume): ").strip()
feature_cols = [c.strip() for c in cols_in.split(",")] if cols_in else None

print("Choose split method:\n1: By Ratio\n2: By Date\n3: By Random")
option = int(input("Select from the above options: "))

ticker = "CBA.AX"

if option == 1:
    test_ratio = float(input("Please enter the test ratio (e.g., 0.2 for 20%): "))
    train_df, test_df, df, scalers = load_data_yf(
        ticker, start, end,
        split_method="ratio", test_size=test_ratio,
        scale=use_scale, feature_cols=feature_cols
    )

elif option == 2:
    split_date = input("Please enter the starting date of test (yyyy-mm-dd): ")
    train_df, test_df, df, scalers = load_data_yf(
        ticker, start, end,
        split_method="date", split_date=split_date,
        scale=use_scale, feature_cols=feature_cols
    )

elif option == 3:
    train_df, test_df, df, scalers = load_data_yf(
        ticker, start, end,
        split_method="random", low=0.15, high=0.30, random_state=314,
        scale=use_scale, feature_cols=feature_cols
    )
else:
    raise ValueError("Invalid option. Choose 1, 2, or 3.")

print(f"No of train rows: {len(train_df)}")
print(f"No of test  rows: {len(test_df)}")
print("Train period:", train_df.index.min().date(), "→", train_df.index.max().date())
print("Test  period:",  test_df.index.min().date(),  "→", test_df.index.max().date())

print("\nTrain head:\n", train_df.head())
print("\nTest head:\n", test_df.head())

# If scaling was enabled, show which columns were scaled
if use_scale:
    print("\nScalers available for columns:", list(scalers.keys()))
