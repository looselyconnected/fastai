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

    kfold_lightgbm(train, test, num_folds=5, params=params, path=PATH, label_col='ID_code', target_col='target')


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
        'early_stopping': 10,
        'lr': 1e-4,
    }

    kfold_cnn(train, test, num_folds=5, params=params, path=PATH, label_col='ID_code', target_col='target',
             name='cnn_model')
    return


def main():
    train = pd.read_csv(f'{PATH}/train.csv')
    test = pd.read_csv(f'{PATH}/test.csv')
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # train_lgb(train, test)
    # train_fc(train, test)
    train_cnn(train, test)

    print('done')


if __name__ == "__main__":
    main()
