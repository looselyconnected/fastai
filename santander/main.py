import numpy as np
import pandas as pd

from os.path import isfile
from scipy.stats import describe

from fastai.structured import *
from fastai.column_data import ColumnarModelData

import matplotlib.pyplot as plt

from common.data import add_stat_features
from common.lgb import kfold_lightgbm

PATH = 'experiments/'


def main():
    train = pd.read_csv(f'{PATH}/train.csv')
    test = pd.read_csv(f'{PATH}/test.csv')

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
        'num_threads': 12,
        'tree_learner': 'serial',
    }

    kfold_lightgbm(train, test, num_folds=5, params=params, path=PATH, label_col='ID_code', target_col='target',
                   )
    print('done')


if __name__ == "__main__":
    main()
