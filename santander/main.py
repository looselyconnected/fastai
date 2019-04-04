import numpy as np
import pandas as pd

from os.path import isfile
from scipy.stats import describe

from fastai.structured import *
from fastai.column_data import ColumnarModelData

import matplotlib.pyplot as plt

from common.data import add_stat_features
from common.lgb import kfold_lightgbm
from common.fc import kfold_fc
from common.cnn import kfold_cnn

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
    params = {
        'emb_drop': 0.0,
        'out_sz': 1,
        'layers': [800],
        'layers_drop': [0.1],
        'epochs': 1000,
        'metrics': ['auc'],
        'binary': True,
        'early_stopping': 10,
        'lr': 3e-4,
    }

    kfold_fc(train, test, num_folds=5, params=params, path=PATH, label_col='ID_code', target_col='target',
             name='fc_model')
    return


def train_cnn(train, test):
    params = {
        'out_sz': 1,
        'layers': [16, 16],
        'layers_drop': [0.1, 0.1],
        'epochs': 1000,
        'metrics': ['auc'],
        'binary': True,
        'early_stopping': 40,
        'lr': 3e-4,
    }

    kfold_cnn(train, test, num_folds=NUM_FOLDS, params=params, path=PATH, label_col='ID_code', target_col='target',
             name='cnn_model', static=STATIC)
    return


def train_secondary(train_df, test_df):
    preds = ['cnn_pred', 'lgb_pred', 'cnn_2feat_pred', 'lgb_5leaf_pred']
    for fn in preds:
        pred_df = pd.read_csv(f'{PATH}/stashed/{fn}.csv').rename(columns={'target': fn})
        train_df = train_df.merge(pred_df.iloc[200000:], on=['ID_code'], how='inner')
        test_df = test_df.merge(pred_df.iloc[:200000], on=['ID_code'], how='inner')

    train_lgb(train_df[['ID_code', 'target'] + preds], test_df[['ID_code'] + preds])
    # train_lgb(train_df.drop(columns=['lgb_pred', 'cnn_pred']), test_df.drop(columns=['lgb_pred', 'cnn_pred']))
    # train_lgb(train_df, test_df)


def main():
    train = pd.read_csv(f'{PATH}/train.csv')
    test = pd.read_csv(f'{PATH}/test.csv')
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Enable the following to train only on half of data
    train_split_idx = int(len(train)/2)
    test = test.append(train.iloc[train_split_idx:], ignore_index=True)
    train = train.iloc[:train_split_idx]

    # train_lgb(train, test)
    # train_fc(train, test)
    train_cnn(train, test)
    # train_secondary(train, test)

    print('done')


if __name__ == "__main__":
    main()
