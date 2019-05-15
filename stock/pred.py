import pandas as pd
import numpy as np
import argparse
import pdb
import re
from os import listdir


from stock.data import Fields as fld, get_ticker_df, index_to_map
from stock.train import *


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

    # If the pred file exists, then only pred the increment
    train_end = int(len(df) * 3 / 5)
    df = df.iloc[train_end:].drop(columns=['target'])

    if args.algo == 'lgb':
        model = LGBModel(f'lgb_{args.by}', path, 'timestamp', 'target', num_folds=5,
                         feat_cols=get_lgb_features(df))
    elif args.algo == 'nn':
        model = NNModel(f'nn_{args.by}', path, 'timestamp', 'target', None, num_folds=1,
                        feat_cols=get_nn_features(df), classification=True, monitor='categorical_accuracy')
    else:
        print('unknown algorithm')
    model.predict(df)
    print('done')


if __name__ == '__main__':
    main()
