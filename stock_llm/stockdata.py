import backoff
import os
import pandas as pd
import numpy as np
import yfinance as yf

class StockData(object):
    # define the columns
    COL_DATE = "Date"
    COL_OPEN = "Open"
    COL_HIGH = "High"
    COL_LOW = "Low"
    COL_CLOSE = "Close"
    COL_ADJ_CLOSE = "Adj Close"
    COL_VOLUME = "Volume"
    COL_DIVIDENDS = "Dividends"
    COL_STOCK_SPLITS = "Stock Splits"
    COL_CAPITAL_GAINS = "Capital Gains"
    COL_DELTA_OPEN = "DeltaOpen"
    COL_DELTA_CLOSE = "DeltaClose"
    COL_DELTA_HIGH = "DeltaHigh"
    COL_DELTA_LOW = "DeltaLow"
    COL_DELTA_VOLUME = "DeltaVolume"
    COL_PREV_CLOSE = "PrevClose"
    COL_PREV_VOLUME = "PrevVolume"

    def __init__(self, symbol: str, path: str):
        self.symbol = symbol
        self.filename = f"{path}/{symbol}.csv"
        self.train_filename = f"{path}/{symbol}_train.csv"
        self.df = None

    # Load the up to date data, returns whether data is loaded from yfinance remotely
    @backoff.on_exception(backoff.expo, Exception, max_time=30)
    def load_yfinance_data(self, use_cache: bool = True) -> bool:
        if os.path.exists(self.filename) and use_cache:
            self.df = pd.read_csv(self.filename)
            if len(self.df) > 100:
                # successfully loaded from disk, return
                return False
            
        ticker = yf.Ticker(self.symbol)
        self.df = ticker.history(period="max", interval="1d", auto_adjust=False).reset_index()
        # save to file
        self.df.to_csv(self.filename, index=False)
        return True

    # Process the data
    def process_data(self):
        # add a column that's the previous's day's close and volume to use as baseline
        self.df[self.COL_PREV_CLOSE] = self.df[self.COL_CLOSE].shift(1)
        self.df[self.COL_PREV_VOLUME] = self.df[self.COL_VOLUME].shift(1)
        self.df[self.COL_DELTA_OPEN] = self.df[self.COL_OPEN] / self.df[self.COL_PREV_CLOSE] - 1
        self.df[self.COL_DELTA_CLOSE] = self.df[self.COL_CLOSE] / self.df[self.COL_PREV_CLOSE] - 1
        self.df[self.COL_DELTA_HIGH] = self.df[self.COL_HIGH] / self.df[self.COL_PREV_CLOSE] - 1
        self.df[self.COL_DELTA_LOW] = self.df[self.COL_LOW] / self.df[self.COL_PREV_CLOSE] - 1
        self.df[self.COL_DELTA_VOLUME] = self.df[self.COL_VOLUME] / self.df[self.COL_PREV_VOLUME] - 1

        # Calculate the std for all the rows before it
        self.df[f"{self.COL_DELTA_OPEN}_std"] = self.df[self.COL_DELTA_OPEN].rolling(200).std()
        self.df[f"{self.COL_DELTA_CLOSE}_std"] = self.df[self.COL_DELTA_CLOSE].rolling(200).std()
        self.df[f"{self.COL_DELTA_HIGH}_std"] = self.df[self.COL_DELTA_HIGH].rolling(200).std()
        self.df[f"{self.COL_DELTA_LOW}_std"] = self.df[self.COL_DELTA_LOW].rolling(200).std()
        self.df[f"{self.COL_DELTA_VOLUME}_std"] = self.df[self.COL_DELTA_VOLUME].rolling(200).std()

        # Bucketize the std column into 0.5*std, 1*std, 2*std buckets and put that into a new column
        bins = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf]
        labels = list(range(0, 8))
        # self.df[f"open_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_OPEN}"]/self.df[f"{self.COL_DELTA_OPEN}_std"], bins=bins, labels=labels)
        self.df[f"close_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_CLOSE}"]/self.df[f"{self.COL_DELTA_CLOSE}_std"], bins=bins, labels=labels)
        self.df[f"high_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_HIGH}"]/self.df[f"{self.COL_DELTA_HIGH}_std"], bins=bins, labels=labels)
        self.df[f"low_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_LOW}"]/self.df[f"{self.COL_DELTA_LOW}_std"], bins=bins, labels=labels)
        self.df[f"volume_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_VOLUME}"]/self.df[f"{self.COL_DELTA_VOLUME}_std"], bins=bins, labels=labels)

        # when this or the prev is invalid, the bucket will be NaN. Fill with the no change bucket - 3
        self.df["close_bucket"] = self.df[f"close_bucket"].fillna(3).astype(int)
        self.df["high_bucket"] = self.df[f"high_bucket"].fillna(3).astype(int)
        self.df["low_bucket"] = self.df[f"low_bucket"].fillna(3).astype(int)
        self.df["volume_bucket"] = self.df[f"volume_bucket"].fillna(3).astype(int)

        self.df['idx'] = np.left_shift(self.df.iloc[200:].close_bucket, 9) | np.left_shift(self.df.iloc[200:].high_bucket, 6) | \
            np.left_shift(self.df.iloc[200:].low_bucket, 3) | self.df.iloc[200:].volume_bucket
        
        # write Date and idx columns out into csv
        self.df.iloc[200:][[self.COL_DATE, 'idx']].to_csv(self.train_filename, index=False)
