from Taskc2 import Loading_and_processing
import mplfinance as fplt 
import numpy as np

# function to create the candlestick chart
def candelstick_plot(ohlc_df,n : int ,type,title ,style ,ylabel ):
    #In here we check if the N value that user enter is greater than the lenth of dataframe that means data rows. 
    #if the n = 5 and there is only 3 rows in dataframe error will be raised.
    if len(ohlc_df) < n:
        raise ValueError(f"Need at least {n} rows to make one candle.")
    # sort it from oldest to newest
    ohlc_df = ohlc_df.sort_index().copy()

    #in here what I have done is I have created a array using numpy to store group of data after deviding it by n value.
    # If n=5 -->[0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2,]
    grp = np.arange(len(ohlc_df)) // n
    # To make a single candle for multiple days we need rules. this tells the pandas to
    #take the first value from “Open” / take the max value from “High” etc.
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
    # plot the data ,has declared the type, title, y axis lable and the style .
    fplt.plot(
        agg,
        type=type,
        title=title,
        style=style,
        ylabel=ylabel
    )

   
    
    
    

if __name__ == "__main__":
    # set inputs for the function.
    ticker = "CBA.AX"
    start = "2020-07-01"
    end   = "2021-07-01"
    test_ratio = 0.2
    use_scale = True
    feature_cols = ["open", "high", "low", "close", "adjclose", "volume"]

    # function which was imported from taskc2.py
    train_df, test_df, df, scalers = Loading_and_processing(
        ticker, start, end,
        split_method="ratio", test_size=test_ratio,
        scale=use_scale, feature_cols=feature_cols
    )
    # to check whether the data is loading correctly from taskc2.py to tskc3.py
    print("Train rows:", len(train_df))
    print("Test rows :", len(test_df))
    print("Full rows :", len(df))
    print(df.head())    # first 5 rows
    print(df.tail())    # last 5 rows

    #mpl finance need the column names capitalized so did rename it.
    ohlc_df = df.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume"

    })
    # ask the user for n value
    ninput=  int(input("Enter Trading period :"))
    #call the candle plot function
    candelstick_plot(ohlc_df,n=ninput, type='candle',title = f"CBA.AX - For 0{ninput} Trading Days" ,style ='charles',ylabel  = 'Price ($)')

    