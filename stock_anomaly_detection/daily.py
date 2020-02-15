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
        df.volume = df.volume.rolling(3).mean()
        add_pct_diff_feature(df, 'adjusted_close', 1)
        add_pct_diff_feature(df, 'adjusted_close', 3)
        add_pct_diff_feature(df, 'adjusted_close', 5)
        add_pct_diff_feature(df, 'volume', 1)
        add_pct_diff_feature(df, 'volume', 3)
        add_pct_diff_feature(df, 'volume', 5)
        add_volatility_feature(df, 'adjusted_close', 0)
        add_volatility_feature(df, 'adjusted_close', 5)
        add_volatility_feature(df, 'volume', 5)

        ticker_dfs.append((ticker, df))

    # merge all the wanted columns into one big df, with the columns prefixed with the ticker name
    concat_dfs = []
    columns_to_use = ['adjusted_close_pct_diff_1',
       'adjusted_close_pct_diff_3', 'adjusted_close_pct_diff_5',
       'volume_pct_diff_1', 'volume_pct_diff_3', 'volume_pct_diff_5',
       'adjusted_close_volatility_0', 'adjusted_close_volatility_5',
       'volume_volatility_5']
    for ticker, df in ticker_dfs:
        new_column_mapping = {x: f'{ticker}_{x}' for x in columns_to_use}
        df = df[['timestamp'] + columns_to_use].rename(columns=new_column_mapping)
        df.set_index(['timestamp'], inplace=True, drop=True)
        concat_dfs.append(df)

    all_df = pd.concat(concat_dfs, axis=1, join='inner').dropna()
    print('done')

    return


if __name__ == '__main__':
    main()
