import subprocess
import os
import argparse
import pandas as pd
import numpy as np

from common.data import get_last_row
from common.feature import add_pct_diff_feature, add_volatility_feature
from stock.data import get_all_data
from stock.pred import predict

def main():
    parser = argparse.ArgumentParser(description='daily run and reporting')
    parser.add_argument("-k", "--key", help="Set the alpha advantage api key")

    args = parser.parse_args()
    if args.key == None:
        print('Must have api key')
        return
    get_all_data('data', args.key)

    index = pd.read_csv(f'data/index.csv')
    ticker_dfs = []  # list of tuples (ticket_name, ticker_df)
    for ticker in index.ticker:
        df = pd.read_csv(f'data/{ticker}.csv')
        add_pct_diff_feature(df, 'adjusted_close', 1)
        add_pct_diff_feature(df, 'adjusted_close', 3)
        add_pct_diff_feature(df, 'adjusted_close', 5)
        add_pct_diff_feature(df, 'volume', 1)
        add_pct_diff_feature(df, 'volume', 3)
        add_pct_diff_feature(df, 'volume', 5)
        add_volatility_feature(df, 'adjusted_close', 5)
        add_volatility_feature(df, 'volume', 5)

        # Add the proxy for today's volatility
        for i in range(0, len(df)):
            ar = [df.loc[i, 'high'], df.loc[i, 'low'], df.loc[i, 'open'], df.loc[i, 'close']]
            df.loc[i, 'adjusted_close_volatility_0'] = np.std(ar) / np.mean(ar)

        ticker_dfs.append((ticker, df))

    # merge all the wanted columns into one big df, with the columns prefixed with the ticker name


    return


if __name__ == '__main__':
    main()
