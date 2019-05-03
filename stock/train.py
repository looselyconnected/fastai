import pandas as pd
import numpy as np
from stock.data import Fields as fld, get_ticker_df, index_to_map
from common.lgb import kfold_lightgbm


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


def add_rank_features(df, index):
    rank_list = []
    index_map = index_to_map(index)

    def get_rank(row):
        result = {}
        for i in range(7):
            result[f'{2**i * 5}'] = []
        ordered_index = row.sort_values().index
        for i in ordered_index:
            split_i = i.split('_')
            result[split_i[2]].append(index_map[split_i[0]])

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


def add_target(df, days, index):
    df_future = df.copy()
    # by attaching the result onto a past row, we look into the future
    df_future.index -= days
    df['target'] = df_future[f'r_{days}_{len(index)}']


def get_all_delta_data(path, index):
    ticker_dfs = {}
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

    return merged


def train(path, target_days):
    index = pd.read_csv(f'{path}/index.csv')
    df = get_all_delta_data(path, index)
    add_rank_features(df, index)
    add_target(df, target_days, index)


    train_end = int(len(df) * 3 / 5)

    params = {
        'boosting': 'gbdt',
        'objective': 'multiclass',
        'num_class': len(index) + 1,  # including cash as a target
        'metric': 'multi_logloss',
        'learning_rate': 0.01,
        'verbose': 1,

        'num_rounds': 10000,
        #'is_unbalance': True,
        #'scale_pos_weight': 8.951238929246692,
        'early_stopping': 500,

        # 'bagging_freq': 5,
        # 'bagging_fraction': 0.33,
        'boost_from_average': 'false',
        'feature_fraction': 1.0,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        # 'min_sum_hessian_in_leaf': 5.0,
        'num_leaves': 5,
        'num_threads': 8,
        'tree_learner': 'serial',
    }

    exclude_cols = ['timestamp', 'target', 'cash_d_5', 'cash_d_10', 'cash_d_20', 'cash_d_40', 'cash_d_80',
                    'cash_d_160', 'cash_d_320']
    for col in df.columns:
        if col.startswith('xl'):
            exclude_cols.append(col)

    test_df = df.iloc[train_end:].drop(columns=['target']).copy()
    kfold_lightgbm(df.iloc[0:train_end], test_df, 5, params, 'data', 'timestamp', 'target', name='lgb',
                   feats_excluded=exclude_cols)


if __name__ == '__main__':
    train(path='data', target_days=160)

    print('done')
