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
from common.lgb import prediction_to_df
from common.nn import split_train_nn


def predict_lgb(path, df, name):
    exclude_cols = ['timestamp', 'target', 'cash_d_5', 'cash_d_10', 'cash_d_20', 'cash_d_40', 'cash_d_80',
                    'cash_d_160', 'cash_d_320']
    for col in df.columns:
        if not col.startswith('r_'):
            exclude_cols.append(col)

    feat_cols = [f for f in df.columns if f not in exclude_cols]

    pred_file = f'{path}/lgb_{name}_pred.csv'
    pred_df = pd.read_csv(pred_file)
    df = df[df.timestamp > pred_df.iloc[-1].timestamp].copy()
    if len(df) == 0:
        return

    sub_preds = None
    num_folds = len([f for f in listdir(f'{path}/models/') if re.match(f'lgb_{name}-[0-9]+', f)])
    for fold in range(num_folds):
        model_name = f'lgb_{name}-{fold}'
        model_path = f'{path}/models/{model_name}'
        model = lgb.Booster(model_file=model_path)
        pred = model.predict(df[feat_cols], num_iteration=model.best_iteration) / num_folds
        if sub_preds is None:
            sub_preds = np.zeros(pred.shape)
        sub_preds += pred

    pred_df = prediction_to_df('target', sub_preds)
    df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    out_cols = ['timestamp'] + pred_df.columns.tolist()
    pred_csv = df[out_cols].to_csv(index=False, header=df.empty)

    f = open(f'{path}/lgb_{name}_pred.csv', 'a')
    f.write(pred_csv)


def predict_nn(path, df, name):
    train_end = int(len(df) * 3 / 5)
    exclude_cols = {'timestamp', 'target', 'cash_d_5', 'cash_d_10', 'cash_d_20', 'cash_d_40', 'cash_d_80',
                    'cash_d_160', 'cash_d_320'}
    for col in df.columns:
        if not col.startswith('r_'):
            exclude_cols.add(col)

    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(len(df.columns) - len(exclude_cols), )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(index) + 1, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(lr=0.0005),
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.categorical_accuracy])

    test_df = df.iloc[train_end:].drop(columns=['target']).copy()
    split_train_nn(model, df.iloc[0:train_end], test_df, path=path, label_col='timestamp', target_col='target',
                   name=f'nn_{name}', target_as_category=True, feats_excluded=exclude_cols, random=True,
                   monitor='categorical_accuracy')


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

    if args.algo == 'lgb':
        predict_lgb(path, df, args.by)
    elif args.algo == 'nn':
        predict_nn(path, df, args.by)
    else:
        print('unknown algorithm')
    print('done')


if __name__ == '__main__':
    main()
