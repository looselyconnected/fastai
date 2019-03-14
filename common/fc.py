import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

import sklearn.datasets
import sklearn.metrics

from common.data import get_embedding_sizes

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from fastai.structured import *
from fastai.column_data import ColumnarModelData


# LightGBM GBDT with KFold or Stratified KFold.
def kfold_fc(train_df, test_df, num_folds, params, path, label_col, target_col,
             feats_excluded=None, out_cols=None, stratified=False, cat_cols=None, name=None):
    print("Starting FC. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    sub_preds = np.zeros(test_df.shape[0])

    if feats_excluded is None:
        feats_excluded = [label_col, target_col]
    feat_cols = [f for f in train_df.columns if f not in feats_excluded]
    if out_cols is None:
        out_cols = [label_col, target_col]
    print(f'features {feat_cols}')

    train_x, train_y, nas, mapper = proc_df(train_df, target_col, do_scale=True, skip_flds=[label_col])
    test_x, _, nas, mapper = proc_df(test_df, target_col, do_scale=True, mapper=mapper, na_dict=nas,
                                     skip_flds=[label_col])
    embedding_sizes = get_embedding_sizes(cat_cols, train_df)
    embedding_inputs = 0
    for embs in embedding_sizes:
        embedding_inputs += embs[1]

    default_layer_size = max(2, int(embedding_sizes ** (1/3)))
    y_range = [train_df[target_col].min(), train_df[target_col].max()]

    lr = params.get('lr', 1e-3)

    # k-fold
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df[feat_cols], train_df[target_col])):
        print("Fold {}".format(fold + 1))

        md = ColumnarModelData.from_data_frame(path, valid_idx, train_x, train_y.astype(np.float32),
                                               cat_flds=cat_cols, is_reg=True, bs=128, test_df=test_x)

        learner = md.get_learner(embedding_sizes, len(train_x.columns) - len(cat_cols),
                                 params.get('emb_drop', 0.1),
                                 params.get('out_sz', 1),
                                 params.get('layers', [default_layer_size ** 2, default_layer_size]),
                                 params.get('layers_drop', [0.3, 0.3]),
                                 y_range=y_range)

        if name is not None:
            try:
                learner.load(name)
            except FileNotFoundError:
                pass

        for i in range(params.get('loops', 10)):
            learner.fit(1e-3, params.get('loop_epoch', 20))
            if name:
                learner.save(name)

        sub_preds += learner.predict(is_test=True) / kf.n_splits

    # save submission file
    test_df.loc[:, target_col] = sub_preds
    test_df = test_df.reset_index()
    test_df[out_cols].to_csv(f'{path}/fc_pred.csv', index=False)
