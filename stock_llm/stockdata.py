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

    BINS = [-np.inf, -3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, np.inf]
    LABELS = np.array(list(range(0, len(BINS)-1)))
    OPEN_LABELS = LABELS
    HIGH_LABELS = OPEN_LABELS + len(LABELS)
    LOW_LABELS = HIGH_LABELS + len(LABELS)
    CLOSE_LABELS = LOW_LABELS + len(LABELS)
    VOLUME_LABELS = CLOSE_LABELS + len(LABELS)

    VIX_BINS = list(range(0, 100, 5)) + [np.inf]
    VIX_LABELS = np.array(list(range(0, len(VIX_BINS)-1))) + VOLUME_LABELS.max() + 1
    LABEL_COUNT = VIX_LABELS.max() + 1

    BIN_NAMES = ['<-3', '-3 to -2','-2 to -1','-1 to -0.5','-0.5 to -0.25','-0.25 to 0','0 to 0.25','0.25 to 0.5','0.5 to 1','1 to 2','2 to 3','>3']  # convert the index of the label into a name
    BIN_VALUES = [-4, -2.5, -1.5, -0.75, -0.375, -0.125, 0.125, 0.375, 0.75, 1.5, 2.5, 3.5, 5]

    # Special contexts that we join to all stock data
    T_VIX = "^VIX"

    @classmethod
    def get_day_divider(cls):
        return cls.LABEL_COUNT

    def __init__(self, symbol: str, path: str):
        self.symbol = symbol
        self.data_path = path
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

    # process the VIX data
    def process_vix_data(self):
        # For VIX data, bin using the raw data instead of delta
        bins = StockData.VIX_BINS
        self.df[f"vix_bucket"] = pd.cut(self.df[self.COL_CLOSE], bins=bins, labels=self.VIX_LABELS)
        self.df.Date = pd.to_datetime(self.df.Date, utc=True).dt.date
        self.df.to_csv(self.train_filename, index=False)

    # Process the data
    def process_data(self):
        if self.symbol == StockData.T_VIX:
            return self.process_vix_data()
        
        self.df.Date = pd.to_datetime(self.df.Date, utc=True).dt.date
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
        bins = StockData.BINS
        self.df[f"open_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_OPEN}"]/self.df[f"{self.COL_DELTA_OPEN}_std"], bins=bins, labels=self.OPEN_LABELS)
        self.df[f"high_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_HIGH}"]/self.df[f"{self.COL_DELTA_HIGH}_std"], bins=bins, labels=self.HIGH_LABELS)
        self.df[f"low_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_LOW}"]/self.df[f"{self.COL_DELTA_LOW}_std"], bins=bins, labels=self.LOW_LABELS)
        self.df[f"close_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_CLOSE}"]/self.df[f"{self.COL_DELTA_CLOSE}_std"], bins=bins, labels=self.CLOSE_LABELS)
        self.df[f"volume_bucket"] = pd.cut(self.df[f"{self.COL_DELTA_VOLUME}"]/self.df[f"{self.COL_DELTA_VOLUME}_std"], bins=bins, labels=self.VOLUME_LABELS)

        # when this or the prev is invalid, the bucket will be NaN. Fill with the no change bucket - 3
        self.df["open_bucket"] = self.df[f"open_bucket"].fillna(np.ceil(self.OPEN_LABELS.mean())).astype(int)
        self.df["high_bucket"] = self.df[f"high_bucket"].fillna(np.ceil(self.HIGH_LABELS.mean())).astype(int)
        self.df["low_bucket"] = self.df[f"low_bucket"].fillna(np.ceil(self.LOW_LABELS.mean())).astype(int)
        self.df["close_bucket"] = self.df[f"close_bucket"].fillna(np.ceil(self.CLOSE_LABELS.mean())).astype(int)
        self.df["volume_bucket"] = self.df[f"volume_bucket"].fillna(np.ceil(self.VOLUME_LABELS.mean())).astype(int)

        # Load vix data and merge 
        vix_df = pd.read_csv(f"{self.data_path}/{self.T_VIX}_train.csv")
        vix_df.Date = pd.to_datetime(vix_df.Date, utc=True).dt.date
        df = pd.merge(self.df.iloc[200:][[self.COL_DATE, 'open_bucket','high_bucket','low_bucket','close_bucket','volume_bucket']], 
                      vix_df[[self.COL_DATE, 'vix_bucket']], how='inner', on=self.COL_DATE)
        df.to_csv(self.train_filename, index=False)
        
        # self.df['idx'] = np.left_shift(self.df.iloc[200:].close_bucket, 9) | np.left_shift(self.df.iloc[200:].high_bucket, 6) | \
        #     np.left_shift(self.df.iloc[200:].low_bucket, 3) | self.df.iloc[200:].volume_bucket
        
        # write Date and label columns out into csv
        # self.df.iloc[200:][[self.COL_DATE, 'open_bucket','high_bucket','low_bucket','close_bucket','volume_bucket']].to_csv(self.train_filename, index=False)