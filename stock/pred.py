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

    index = pd.read_csv(f'data/index_by_{args.by}.csv')
    predict('data', args.algo, args.by, index)


def predict(path, algo, by, index):
    df = get_all_delta_data(path, index)
    add_rank_features(df, index)
    df = add_put_call_features(df, path)

    # If the pred file exists, then only pred the increment
    train_end = int(len(df) * 3 / 5)
    df = df.iloc[train_end:]

    if algo == 'lgb':
        model = LGBModel(f'lgb_{by}', path, 'timestamp', 'target', num_folds=5,
                         feat_cols=get_lgb_features(df))
    elif algo == 'nn':
        model = NNModel(f'nn_{by}', path, 'timestamp', 'target', None, num_folds=1,
                        feat_cols=get_nn_features(df), classification=True, monitor='categorical_accuracy')
    else:
        print('unknown algorithm')
    model.predict(df)
    print('done')


if __name__ == '__main__':
    main()
