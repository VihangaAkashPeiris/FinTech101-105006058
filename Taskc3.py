from Taskc2 import Loading_and_processing
import mplfinance as fplt 
import matplotlib.pyplot as plt
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
    # use the first date of each n-day block 
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
# function to create the boxplot chart
def boxplot(daily_df, n):
    daily_df = daily_df.sort_index().copy() #sort data frame  from old to newest
    #in here what I have done is I have created a array using numpy to store group of data after deviding it by n value.
    # If n=5 -->[0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2,]
    groups = np.arange(len(daily_df)) // n 
    # This will group the data and will get the close value of each date in that group and make a array.
    data   = [g['Close'].values for _, g in daily_df.groupby(groups)] 

    # I use 'block_days' just to store labels for x axis. In here it gets the first date of each group as label.
    block_dates = [daily_df.index[i*n] for i in range(len(data))]
    #set the plot size
    plt.figure(figsize=(12,6))
    # this gets the data that need to map the boxplot and 'showmeans=true' will show a marker that shows mean.
    plt.boxplot(data, showmeans=True)

    # Here I have arranged the x axis labels. It will have ticks according to the length of data groups and will write the date as a yyyy-mm-dd in string format
    plt.xticks(ticks=np.arange(1, len(block_dates)+1), 
               labels=[d.strftime('%Y-%m-%d') for d in block_dates], 
               rotation=45, ha='right') # this do rotate the label and also position the label at the right side of the edge tick.

    # 'ax' will have the axes of the boxplot and we can manipulate those axes from ax.
    ax = plt.gca()
    #This was basically done to avoid label's overlapping 
    # iF there are a lots of ticks labels miight overlap. to avoid that we are only show every 5th lable
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 5 != 0:   # this ensures that it shows every 5th lable.
            label.set_visible(False)

    plt.title(f' CBA.AX : Close distribution in each {n}-day block') # title of the plot
    plt.ylabel('Price ($)') # y axis title
    plt.grid(True, axis='y', alpha=.3)  # this will make the horizontal gridline visible with 30% opacity.
    plt.tight_layout() # will adjust the frame according to labels. so we don't loose any lables.
    plt.show() # adjust margins so labels/titles don’t get cut off.

if __name__ == "__main__":
    # set inputs for the function.
    ticker = "CBA.AX"
    start = "2018-01-01"
    end   = "2021-12-31"
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
    ninput=  int(input("Enter Trading period days :"))
    print("Chart types -")
    print("01. Candlestick Chart\n02. Boxplot Chart")
    charttype = int(input("What chart would You prefer (enter the no)-"))


    if (charttype==1):
    #call the candle plot function
        candelstick_plot(ohlc_df,n=ninput, type='candle',
                     title = f"CBA.AX  Candelstick Chart - For  {ninput} Trading Days" ,
                     style ='charles',ylabel  = 'Price ($)')
    else:
    # call the box plot function
        boxplot(ohlc_df,n=ninput)
    