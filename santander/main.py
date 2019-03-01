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
        'metric': 'binary',
        'learning_rate': 0.01,
        'num_rounds': 20000,
        'is_unbalance': True,
        'verbose': 1,
    }

    kfold_lightgbm(train, test, num_folds=5, params=params, path=PATH, label_col='ID_code', target_col='target')
    print('done')


if __name__ == "__main__":
    main()
