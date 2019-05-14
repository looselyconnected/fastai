import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

import sklearn.datasets
import sklearn.metrics
import optuna

from common.data import prediction_to_df, display_importances
from common.model import *
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


class LGBModel(MLModel):
    def __init__(self, name, path, label_col, target_col, num_folds=0, feat_cols=None, out_cols=None):
        super().__init__(name, path, label_col, target_col, num_folds, feat_cols, out_cols)
        self.feature_importance_df = pd.DataFrame()

    def load(self, model_path):
        self.model = lgb.Booster(model_file=model_path)

    def save(self, model_path):
        self.model.save_model(filename=model_path, num_iteration=self.model.best_iteration)
        print(f'saved model iteration {self.model.best_iteration} to {model_path}')

    def train_one_fold(self, fold, params, train_df, train_idx, valid_idx):
        params['seed'] = params['bagging_seed'] = params['drop_seed'] = int(2 ** fold)
        train_set = lgb.Dataset(train_df[self.feat_cols].iloc[train_idx],
                                label=train_df[self.target_col].iloc[train_idx])
        valid_set = lgb.Dataset(train_df[self.feat_cols].iloc[valid_idx],
                                label=train_df[self.target_col].iloc[valid_idx])

        self.model = lgb.train(params, train_set, valid_sets=valid_set, verbose_eval=100)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = self.feat_cols
        fold_importance_df["importance"] = np.log1p(
            self.model.feature_importance(importance_type='gain', iteration=self.model.best_iteration))
        fold_importance_df["fold"] = fold + 1
        self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)

    def predict_one_fold(self, df):
        return self.model.predict(df[self.feat_cols], num_iteration=self.model.best_iteration)

    def post_train(self):
        super().post_train()
        display_importances(self.feature_importance_df)


# LightGBM GBDT with KFold or Stratified KFold.
def lgb_train(train_df, num_folds, params, path, label_col, target_col,
              feats_excluded=None, stratified=False, name=None, static=False):
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))

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


def lgb_predict(df, num_folds, path, label_col, target_col,
                feats_excluded=None, out_cols=None, name=None):
    if feats_excluded is None:
        feats_excluded = [label_col, target_col]
    feat_cols = [f for f in df.columns if f not in feats_excluded]
    print(f'features {feat_cols}')

    include_header = False
    pred_file = f'{path}/{name}_pred.csv'
    try:
        last_pred_time = pd.read_csv(pred_file).iloc[-1].timestamp
        df = df[df.timestamp > last_pred_time].copy()
        if len(df) == 0:
            return
    except:
        include_header = True

    sub_preds = None
    for fold in range(num_folds):
        model_name = f'{name}-{fold}'
        model_path = f'{path}/models/{model_name}'

        model = lgb.Booster(model_file=model_path)
        pred = model.predict(df[feat_cols], num_iteration=model.best_iteration) / num_folds
        if sub_preds is None:
            sub_preds = np.zeros(pred.shape)
        sub_preds += pred

    pred_df = prediction_to_df(target_col, sub_preds)
    df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    if out_cols is None:
        out_cols = [label_col] + pred_df.columns.tolist()

    if include_header:
        df[out_cols].to_csv(pred_file, index=False)
    else:
        out_csv = df[out_cols].to_csv(index=False, header=include_header)
        f = open(pred_file, 'a')
        f.write(out_csv)


def lgb_params_tune(train_df, test_df, params, label_col, target_col,
                    feats_excluded=None, out_cols=None):

    return