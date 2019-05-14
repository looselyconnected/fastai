import pandas as pd
import numpy as np
import argparse
import pdb
import re
from tensorflow.python import keras
from tensorflow.python.keras import optimizers
from os import listdir

import lightgbm as lgb


from stock.data import Fields as fld, get_ticker_df, index_to_map
from stock.train import *
from common.lgb import lgb_predict
from common.nn import split_train_nn


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

    exclude_cols = ['timestamp', 'target', 'cash_d_5', 'cash_d_10', 'cash_d_20', 'cash_d_40', 'cash_d_80',
                    'cash_d_160', 'cash_d_320']
    for col in df.columns:
        if not col.startswith('r_'):
            exclude_cols.append(col)
    feat_cols = [f for f in df.columns if f not in exclude_cols]

    # If the pred file exists, then only pred the increment
    train_end = int(len(df) * 3 / 5)
    df = df.iloc[train_end:].drop(columns=['target'])

    if args.algo == 'lgb':
        model = LGBModel(f'lgb_{args.by}', path, 'timestamp', 'target', num_folds=5,
                         feat_cols=feat_cols)
    elif args.algo == 'nn':
        model = NNModel(f'nn_{args.by}', path, 'timestamp', 'target', None, num_folds=1,
                        feat_cols=feat_cols, classification=True, monitor='categorical_accuracy')
    else:
        print('unknown algorithm')
    model.predict(df)
    print('done')


if __name__ == '__main__':
    main()
