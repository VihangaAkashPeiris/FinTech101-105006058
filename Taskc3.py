from Taskc2 import Loading_and_processing
import mplfinance as fplt 
import numpy as np
def candelstick_plot(ohlc_df,n : int ,type,title ,style ,ylabel ):



    
    

    if len(ohlc_df) < n:
        raise ValueError(f"Need at least {n} rows to make one candle.")
    
    ohlc_df = ohlc_df.sort_index().copy()
    
    grp = np.arange(len(ohlc_df)) // n
    agg = ohlc_df.groupby(grp).agg({
        "Open":  "first",
        "High":  "max",
        "Low":   "min",
        "Close": "last",
        "Volume":"sum",
    })

    # Give a meaningful datetime index to the compressed candles:
    # use the first date of each n-day block (you can also pick the last date if you prefer)
    idx = ohlc_df.index[::n][:len(agg)]
    agg.index = idx
    
    fplt.plot(
        agg,
        type=type,
        title=title,
        style=style,
        ylabel=ylabel
    )

   
    
    
    

if __name__ == "__main__":
    # ----- set inputs -----
    ticker = "CBA.AX"
    start = "2020-07-01"
    end   = "2021-07-01"
    test_ratio = 0.2
    use_scale = True
    feature_cols = ["open", "high", "low", "close", "adjclose", "volume"]

    # ----- call function from Taskc2 -----
    train_df, test_df, df, scalers = Loading_and_processing(
        ticker, start, end,
        split_method="ratio", test_size=test_ratio,
        scale=use_scale, feature_cols=feature_cols
    )
    # ----- quick checks -----
    print("Train rows:", len(train_df))
    print("Test rows :", len(test_df))
    print("Full rows :", len(df))
    print(df.head())    # first 5 rows
    print(df.tail())    # last 5 rows
    ohlc_df = df.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume"

    })
    ninput=  int(input("Enter Trading period :"))

    candelstick_plot(ohlc_df,n=ninput, type='candle',title = f"CBA.AX - For 0{ninput} Trading Days" ,style ='charles',ylabel  = 'Price ($)')

    