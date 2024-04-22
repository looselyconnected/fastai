import argparse
import os
import pandas as pd
import random
import time

from stockdata import StockData


def get_ticker_data(ticker: str, path: str, use_cache: bool = True) -> bool:
    sd = StockData(ticker, path)
    loaded_from_yfinance = sd.load_yfinance_data(use_cache=use_cache)
    sd.process_data()
    return loaded_from_yfinance

def get_all_data(path: str, use_cache: bool = True):
    index = pd.read_csv(f"{path}/index.csv")
    try:
        index = pd.read_csv(f"{path}/index.csv")
    except:
        print("create index.csv first")

    for ticker in index.ticker:
        loaded_from_yfinance = get_ticker_data(ticker, path, use_cache)
        if loaded_from_yfinance:
            # Must slow down to avoid throttling by API server
            time.sleep(random.randint(1, 5))

def main():
    currentDir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description="downloading data and process into train.csv")
    parser.add_argument("-o", "--output", help="Set output directory name.", default="data")
    parser.add_argument("-c", "--use_cache", help="Use Cache", default=True)
    parser.add_argument("-t", "--ticker", help="Stock ticker (e.g. SPY)", default=None)

    args = parser.parse_args()

    path = f"{currentDir}/{args.output}"
    if not os.path.exists(path):
        os.makedirs(path)

    if args.ticker:
        get_ticker_data(args.ticker, path, args.use_cache)
        return

    get_all_data(path, args.use_cache)

if __name__ == "__main__":
    main()
