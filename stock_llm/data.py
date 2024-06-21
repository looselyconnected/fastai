import argparse
import numpy as np
import os
import torch
import pandas as pd
import random
import time

from typing import Tuple

from stockdata import StockData

data_columns = ["close_direction", "close_bucket", "volume_bucket", "vix_bucket", "tnx_bucket", "open_bucket", "high_bucket", "low_bucket", "divider"]

def get_ticker_data(ticker: str, path: str, intra_day: bool, use_cache: bool = True) -> Tuple[StockData, bool]:
    sd = StockData(ticker, path, intra_day)
    loaded_from_yfinance = sd.load_yfinance_data(use_cache=use_cache)
    sd.process_data()
    return (sd, loaded_from_yfinance)

def get_all_data(path: str, intra_day: bool, use_cache: bool = True):
    index = pd.read_csv(f"{path}/index.csv")
    try:
        index = pd.read_csv(f"{path}/index.csv")
    except:
        print("create index.csv first")

    for ticker in index.ticker:
        _, loaded_from_yfinance = get_ticker_data(ticker, path, intra_day=intra_day, use_cache=use_cache)
        if loaded_from_yfinance:
            # Must slow down to avoid throttling by API server
            time.sleep(random.randint(1, 5))

def generate_train_data(data_dir: str, train_cutoff_date: str):
    row_divider = StockData.ROW_DIVIDER
    stock_divider = StockData.STOCK_DIVIDER

    # load in all the data. Split train and val on date to avoid leakage
    train_data = np.array([], dtype=np.int16)
    val_data = np.array([], dtype=np.int16)
    # read in the *_train.csv files
    for f in os.listdir(data_dir):
        if f.endswith('_train.csv') and not f.startswith('^'):
            df = pd.read_csv(f"{data_dir}/{f}")
            if len(df) > 1024:
                # add a divider column, that contains just the special divider char
                df['divider'] = row_divider
                train_data = np.append(train_data, df.loc[df.Date < train_cutoff_date, data_columns].values.flatten())
                train_data = np.append(train_data, [stock_divider]*5) 
                val_data = np.append(val_data, df.loc[df.Date >= train_cutoff_date, data_columns].values.flatten())
                val_data = np.append(val_data, [stock_divider]*5) 

    # write the train and val data into file
    np.savez(f"{data_dir}/train_eval.npz", train=train_data, val=val_data)

# encode dataframe into token sequences
def encode_data(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(np.array([df[data_columns].values.astype(np.int32).flatten()]))

# Decode the prediction tensor back to a df with data_columns
def decode_data(t: torch.Tensor) -> pd.DataFrame:
    df = pd.DataFrame(t.cpu().view(-1, len(data_columns)), columns=data_columns)
    return df

def label_predictions(df: pd.DataFrame):
    df['open'] = df['open_bucket'].map(lambda x: StockData.BIN_NAMES[x])
    df['high'] = df['high_bucket'].map(lambda x: StockData.BIN_NAMES[x])
    df['low'] = df['low_bucket'].map(lambda x: StockData.BIN_NAMES[x])
    df['close'] = df['close_bucket'].map(lambda x: StockData.BIN_NAMES[x])
    df['volume'] = df['volume_bucket'].map(lambda x: StockData.BIN_NAMES[x])

def get_data_for_eval(ticker: str, data_dir: str, intra_day: bool = False) -> pd.DataFrame:
    df = pd.read_csv(f"{data_dir}/{ticker}_train.csv")
    df['divider'] = StockData.ROW_DIVIDER
    if intra_day:
        df.Date = pd.to_datetime(df.Date, utc=True)
    else:
        df.Date = pd.to_datetime(df.Date, utc=True).dt.date
    return df

def main():
    currentDir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description="downloading data and process into train.csv")
    parser.add_argument("-o", "--output", help="Set output directory name.", default=None)
    parser.add_argument("-c", "--use_cache", help="Use Cache", action="store_true", default=False)
    parser.add_argument("-t", "--ticker", help="Stock ticker (e.g. SPY)", default=None)
    parser.add_argument("-i", "--intra_day", help="Get intra-day data", action="store_true", default=False)

    args = parser.parse_args()

    cutoff_date = '2020-01-01'
    default_output = "data"
    if args.intra_day:
        # Get the date that's today - 15 days
        cutoff_date = (pd.to_datetime(pd.Timestamp.now()) - pd.Timedelta(days=15)).strftime('%Y-%m-%d')
        default_output = "data_intra_day"
    if args.output is None:
        args.output = default_output

    path = f"{currentDir}/{args.output}"
    if not os.path.exists(path):
        os.makedirs(path)

    if args.ticker:
        get_ticker_data(args.ticker, path, args.intra_day, args.use_cache)
        # generate_train_data(path, cutoff_date)
        return

    get_all_data(path, args.use_cache)
        
    generate_train_data(path, cutoff_date)

if __name__ == "__main__":
    main()
