import pandas as pd
import numpy as np
import argparse
import pdb
from tensorflow.python import keras
from tensorflow.python.keras import optimizers

from stock.data import Fields as fld, get_ticker_df, index_to_map
from common.lgb import LGBModel
from common.nn import NNModel


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


def get_lgb_features(df):
    exclude_cols = ['timestamp', 'target', 'cash_d_5', 'cash_d_10', 'cash_d_20', 'cash_d_40', 'cash_d_80',
                    'cash_d_160', 'cash_d_320']
    for col in df.columns:
        if not col.startswith('r_') or col.startswith('r_5') or col.startswith('r_10'):
            exclude_cols.append(col)
    return [f for f in df.columns if f not in exclude_cols]


def get_nn_features(df):
    exclude_cols = {'timestamp', 'target', 'cash_d_5', 'cash_d_10', 'cash_d_20', 'cash_d_40', 'cash_d_80',
                    'cash_d_160', 'cash_d_320'}
    for col in df.columns:
        if col.startswith('r_') or col.endswith('_d_5') or col.endswith('_d_10'):
            exclude_cols.add(col)
    return [f for f in df.columns if f not in exclude_cols]


def train_lgb(path, index, df, name):
    params = {
        'boosting': 'gbdt',
        'objective': 'multiclass',
        'num_class': len(index) + 1,  # including cash as a target
        'metric': 'multi_logloss',
        'learning_rate': 0.005,
        'verbose': 1,

        'num_rounds': 30000,
        #'is_unbalance': True,
        #'scale_pos_weight': 8.951238929246692,
        'early_stopping': 1000,

        # 'bagging_freq': 5,
        # 'bagging_fraction': 0.33,
        'boost_from_average': 'false',
        'feature_fraction': 1.0,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        # 'min_sum_hessian_in_leaf': 5.0,
        'num_leaves': 3,
        'num_threads': 8,
        'tree_learner': 'serial',
    }


    lgb_model = LGBModel(f'lgb_{name}', path, 'timestamp', 'target', num_folds=5,
                         feat_cols=get_lgb_features(df))
    lgb_model.train(df, params, stratified=False, random_shuffle=True)


def train_nn(path, index, df, name):
    feat_cols = get_nn_features(df)
    model = keras.models.Sequential([
        keras.layers.Dense(1024, activation='relu', input_shape=(len(feat_cols), ),
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        # keras.layers.Dense(64, activation='relu'),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(len(index) + 1, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(lr=0.0005),
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.categorical_accuracy])
    nn_model = NNModel(f'nn_{name}', path, 'timestamp', 'target', model, num_folds=1,
                       feat_cols=feat_cols, classification=True, monitor='categorical_accuracy')
    nn_model.train(df, None)


def main():
    parser = argparse.ArgumentParser(description='testing performance')
    parser.add_argument("-a", "--algo", help="The algorithm we want to test ")
    parser.add_argument("-b", "--by", help="The market segment breakdown method, sector or size")

    args = parser.parse_args()
    if args.algo is None:
        print('Must have algo name')
        return
    if args.by is None:
        print('Must specify -b sector or size')
        return

    path = 'data'

    index = pd.read_csv(f'{path}/index_by_{args.by}.csv')
    df = get_all_delta_data(path, index)
    add_rank_features(df, index)
    add_target(df, 160, index)

    train_end = int(len(df) * 3 / 5)
    df = df.iloc[0:train_end]
    if args.algo == 'lgb':
        train_lgb(path, index, df, args.by)
    elif args.algo == 'nn':
        train_nn(path, index, df, args.by)
    else:
        print('unknown algorithm')
    print('done')


if __name__ == '__main__':
    main()
