import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

import sklearn.datasets
import sklearn.metrics
import optuna

from common.data import display_importances
from common.data import rmse
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


# LightGBM GBDT with KFold or Stratified KFold.
def kfold_lightgbm(train_df, test_df, num_folds, params, path, label_col, target_col,
                   feats_excluded=None, out_cols=None, stratified=False, name=None, static=False):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape,
                                                                      test_df.shape if test_df is not None else 0))

    # Cross validation model
    if stratified:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    sub_preds = None
    feature_importance_df = pd.DataFrame()

    if feats_excluded is None:
        feats_excluded = [label_col, target_col]
    feat_cols = [f for f in train_df.columns if f not in feats_excluded]
    print(f'features {feat_cols}')

    if static:
        num_folds = 1
    # k-fold
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df[feat_cols], train_df[target_col])):
        print("Fold {}".format(fold + 1))
        model_name = f'{name}-{fold}'
        model_path = f'{path}/models/{model_name}'

        params['seed'] = params['bagging_seed'] = params['drop_seed'] = int(2 ** fold)
        train_set = lgb.Dataset(train_df[feat_cols].iloc[train_idx], label=train_df[target_col].iloc[train_idx])
        valid_set = lgb.Dataset(train_df[feat_cols].iloc[valid_idx], label=train_df[target_col].iloc[valid_idx])

        model = lgb.train(params, train_set, valid_sets=valid_set, verbose_eval=100)
        model.save_model(filename=model_path, num_iteration=model.best_iteration)
        if test_df is not None:
            pred = model.predict(test_df[feat_cols], num_iteration=model.best_iteration) / num_folds
            if sub_preds is None:
                sub_preds = np.zeros(pred.shape)
            sub_preds += pred

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feat_cols
        fold_importance_df["importance"] = np.log1p(
            model.feature_importance(importance_type='gain', iteration=model.best_iteration))
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        if static:
            break

    # display importances
    display_importances(feature_importance_df)

    # save submission file
    if test_df is not None:
        pred_df = prediction_to_df(target_col, sub_preds)

        test_df = test_df.reset_index()
        test_df = pd.concat([test_df, pred_df], axis=1)

        if out_cols is None:
            out_cols = [label_col] + pred_df.columns
        test_df[out_cols].to_csv(f'{path}/{name}_pred.csv', index=False)


def prediction_to_df(target_col, pred):
    if len(pred.shape) == 2:
        pred_cols = [f'{target_col}_{i}' for i in range(pred.shape[1])]
    else:
        pred_cols = [target_col]
    return pd.DataFrame(pred, columns=pred_cols)


def lgb_params_tune(train_df, test_df, params, label_col, target_col,
                    feats_excluded=None, out_cols=None):

    return