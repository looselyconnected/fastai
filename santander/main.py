import numpy as np
import pandas as pd

from os.path import isfile
from scipy.stats import describe

from tensorflow.python import keras

from fastai.structured import *
from fastai.column_data import ColumnarModelData

import matplotlib.pyplot as plt

from common.data import add_stat_features
from common.lgb import kfold_lightgbm
from common.fc import kfold_fc
from common.cnn import kfold_cnn
from common.nn import kfold_nn

PATH = 'experiments/'

NUM_FOLDS = 5
STATIC = False

def train_lgb(train, test):
    # feature_cols = list(set(train.columns) - set(['ID_code', 'target']))
    # add_stat_features(train, feature_cols)
    # add_stat_features(test, feature_cols)
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'verbose': 1,

        'num_rounds': 30000,
        #'is_unbalance': True,
        #'scale_pos_weight': 8.951238929246692,
        'early_stopping': 3000,

        'bagging_freq': 5,
        'bagging_fraction': 0.33,
        'boost_from_average': 'false',
        'feature_fraction': 0.05,
        'max_depth': -1,
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'tree_learner': 'serial',
    }

    kfold_lightgbm(train, test, num_folds=NUM_FOLDS, params=params, path=PATH,
                   label_col='ID_code', target_col='target', static=STATIC)


def train_fc(train, test):
    model = keras.models.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(4, )),
        # keras.layers.Dropout(0.1),
        keras.layers.Dense(16, activation='relu'),
        # keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[keras.metrics.binary_accuracy])

    kfold_nn(model, train, test, num_folds=NUM_FOLDS, path=PATH, label_col='ID_code', target_col='target',
             name='fc_model')
    return


def train_cnn(train, test):
    model = keras.models.Sequential([
        keras.layers.Conv1D(16, 3, padding='same', activation='relu', input_shape=(200, 1)),
        keras.layers.MaxPool1D(2),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2, padding='same'),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2, padding='same'),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2, padding='same'),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(16, 3, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2, padding='same'),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(8, 3, padding='same', activation='relu'),
        keras.layers.MaxPool1D(2, padding='same'),
        keras.layers.Dropout(0.1),
        keras.layers.Flatten(),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[keras.metrics.binary_accuracy])

    kfold_nn(model, train, test, num_folds=NUM_FOLDS, path=PATH, label_col='ID_code', target_col='target',
             name='cnn_model', input_shape=(200, 1))
    return


def train_secondary(train_df, test_df):
    preds = ['cnn_pred', 'lgb_pred', 'lgb_5leaf_pred', 'cnn_16feat_tf_pred']
    for fn in preds:
        pred_df = pd.read_csv(f'{PATH}/stashed/{fn}.csv').rename(columns={'target': fn})
        train_df = train_df.merge(pred_df.iloc[200000:], on=['ID_code'], how='inner')
        test_df = test_df.merge(pred_df.iloc[:200000], on=['ID_code'], how='inner')

    train_fc(train_df[['ID_code', 'target'] + preds], test_df[['ID_code'] + preds])
    # train_lgb(train_df.drop(columns=['lgb_pred', 'cnn_pred']), test_df.drop(columns=['lgb_pred', 'cnn_pred']))
    # train_lgb(train_df, test_df)


def main():
    train = pd.read_csv(f'{PATH}/train.csv')
    test = pd.read_csv(f'{PATH}/test.csv')
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Enable the following to train only on half of data
    # train_split_idx = int(len(train)/2)
    # test = test.append(train.iloc[train_split_idx:], ignore_index=True, sort=False)
    # train = train.iloc[:train_split_idx]

    # train_lgb(train, test)
    # train_fc(train, test)
    # train_cnn(train, test)
    train_secondary(train, test)

    print('done')


if __name__ == "__main__":
    main()
