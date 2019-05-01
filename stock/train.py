import pandas as pd
import numpy as np
from stock.data import Fields as fld
from common.lgb import kfold_lightgbm


def get_ticker_df(path, ticker, cols=None):
    return pd.read_csv(f'{path}/{ticker}.csv', usecols=cols)


def get_delta_columns(prefix):
    return [f'{prefix}_d_{2**i * 5}' for i in range(7)]


def add_ticker_features(df, prefix):
    df_copy = df.copy()

    # add 5, 10, 20, 40, 80, 160, 320 day delta
    for i in range(7):
        days_interval = 2**i * 5
        df_copy.index += days_interval
        df[f'{prefix}_d_{days_interval}'] = df[fld.COL_ADJ_CLOSE] / df_copy[fld.COL_ADJ_CLOSE] - 1
        df_copy.index -= days_interval

    df.dropna(inplace=True)


def add_cash_features(df):
    # add 5, 10, 20, 40, 80, 160, 320 day delta
    for i in range(7):
        days_interval = 2**i * 5
        df[f'cash_d_{days_interval}'] = 0.0


def add_rank_features(df):
    rank_list = []

    def get_rank(row):
        result = {}
        for i in range(7):
            result[f'{2**i * 5}'] = []
        ordered_index = row.sort_values().index
        for i in ordered_index:
            split_i = i.split('_')
            result[split_i[2]].append(split_i[0])

        flat = []
        for i in range(7):
            flat += result[f'{2**i * 5}']
        rank_list.append(flat)

    df.drop(fld.COL_TIME, axis=1).apply(get_rank, axis=1)

    ticker_count = int(len(rank_list[0]) / 7)
    ar = np.array(rank_list)
    assert(len(df) == len(ar))
    new_col_list = [f'r_{2**i * 5}_{j}' for i in range(7) for j in range(ticker_count)]

    df[new_col_list] = pd.DataFrame(ar, columns=new_col_list)
    return


def add_target(df, days):
    df_future = df.copy()
    # by attaching the result onto a past row, we look into the future
    df_future.index -= days
    df['target'] = df_future[f'r_{days}_9']
    df.dropna(inplace=True)


def get_all_delta_data(path):
    ticker_dfs = {}
    index = pd.read_csv(f'{path}/index.csv')
    for ticker in index.ticker:
        df = get_ticker_df(path, ticker)
        add_ticker_features(df, ticker)
        ticker_dfs[ticker] = df

    # merge just the delta columns
    merged = None
    for ticker, v in ticker_dfs.items():
        if merged is None:
            merged = v[[fld.COL_TIME] + get_delta_columns(ticker)]
        else:
            merged = pd.merge(merged, v[[fld.COL_TIME] + get_delta_columns(ticker)], on=fld.COL_TIME)

    add_cash_features(merged)

    add_rank_features(merged)
    add_target(merged, 80)

    return merged


def train(path):
    df = get_all_delta_data(path)



if __name__ == '__main__':
    train(path='data')

    print('done')
