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
                   feats_excluded=None, out_cols=None, stratified=False):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    sub_preds = np.zeros(test_df.shape[0])
    train_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    if feats_excluded is None:
        feats_excluded = [label_col, target_col]
    feat_cols = [f for f in train_df.columns if f not in feats_excluded]
    if out_cols is None:
        out_cols = [label_col, target_col]
    print(f'features {feat_cols}')

    # k-fold
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df[feat_cols], train_df[target_col])):
        print("Fold {}".format(fold + 1))
        params['seed'] = params['bagging_seed'] = params['drop_seed'] = int(2 ** fold)
        train_set = lgb.Dataset(train_df[feat_cols].iloc[train_idx], label=train_df[target_col].iloc[train_idx])
        valid_set = lgb.Dataset(train_df[feat_cols].iloc[valid_idx], label=train_df[target_col].iloc[valid_idx])

        model = lgb.train(params, train_set, valid_sets=valid_set, verbose_eval=100)
        sub_preds += model.predict(test_df[feat_cols], num_iteration=model.best_iteration) / kf.n_splits
        train_preds += model.predict(train_df[feat_cols], num_iteration=model.best_iteration) / kf.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feat_cols
        fold_importance_df["importance"] = np.log1p(
            model.feature_importance(importance_type='gain', iteration=model.best_iteration))
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    # display importances
    display_importances(feature_importance_df)

    # save submission file
    test_df.loc[:, target_col] = sub_preds
    test_df = test_df.reset_index()
    test_df[out_cols].to_csv(f'{path}/lgb_pred.csv', index=False)

    # save the result for the training file
    train_df_pred = train_df[out_cols].copy()
    train_df_pred['lgb_pred'] = train_preds
    train_df_pred.to_csv(f'{path}/lgb_train_pred.csv', index=False)


def lgb_params_tune(train_df, test_df, params, label_col, target_col,
                    feats_excluded=None, out_cols=None):

    return