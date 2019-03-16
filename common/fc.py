import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

import sklearn.datasets

from common.data import get_embedding_sizes, get_validation_index

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from fastai.structured import *
from fastai.metrics import *
from fastai.column_data import ColumnarModelData
from fastai.sgdr import LossRecorder


def roc_auc(preds, y_true):
    return metrics.roc_auc_score(y_true, preds)


metrics_map = {
    'auc': roc_auc,
    'accuracy': accuracy,
    'f1': f1,
}


class SaveBestModel(LossRecorder):
    def __init__(self, model, lr, name, early_stopping=0):
        super().__init__(model.get_layer_opt(lr, None))
        assert(name is not None)
        self.name = name
        self.model = model
        self.best_loss = 1e20
        self.best_epoch = 0
        self.early_stopping = early_stopping

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        loss = metrics[0][-1]
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = self.epoch
            self.model.save(self.name)
        elif self.early_stopping > 0 and self.epoch >= self.best_epoch + self.early_stopping:
            print(f'Early stopping after loss not improving after {self.early_stopping} epochs')
            return True


# LightGBM GBDT with KFold or Stratified KFold.
def kfold_fc(train_df, test_df, num_folds, params, path, label_col, target_col,
             feats_excluded=None, out_cols=None, stratified=False, cat_cols=[], name=None):
    print("Starting FC. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    train_df[target_col] = train_df[target_col].astype(float)
    # Cross validation model
    if stratified:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    test_df[target_col] = 0

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

    default_layer_size = max(2, int((embedding_inputs + len(train_x.columns) - len(cat_cols)) ** (1/3)))
    y_range = [train_df[target_col].min(), train_df[target_col].max()]

    lr = params.get('lr', 1e-3)
    train_metrics = None
    param_metrics = params.get('metrics')
    if param_metrics is not None:
        train_metrics = []
        for metric in param_metrics:
            train_metrics.append(metrics_map[metric])

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df[feat_cols], train_df[target_col])):
        print("Fold {}".format(fold + 1))
        model_name = f'{name}-{fold}'

        md = ColumnarModelData.from_data_frame(path, valid_idx, train_x, train_y.astype(np.float32),
                                               cat_flds=cat_cols, is_reg=True, bs=128, test_df=test_x)

        learner = md.get_learner(embedding_sizes, len(train_x.columns) - len(cat_cols),
                                 params.get('emb_drop', 0.1),
                                 params.get('out_sz', 1),
                                 params.get('layers', [default_layer_size ** 2, default_layer_size]),
                                 params.get('layers_drop', [0.3, 0.3]),
                                 metrics=train_metrics,
                                 y_range=y_range)
        callback = SaveBestModel(learner, lr, model_name, params.get('early_stopping', 0))

        if params.get('binary', False):
            learner.crit = F.binary_cross_entropy

        if name is not None:
            try:
                learner.load(model_name)
            except FileNotFoundError:
                pass

        learner.fit(lr, params.get('epochs', 20), callbacks=[callback])

        # load the best model
        print(f'Best epoch is {callback.best_epoch} loss {callback.best_loss}')
        learner.load(model_name)
        test_df.loc[:, target_col] += (learner.predict(is_test=True).reshape(len(test_df)) / kf.n_splits)

        # save submission file
    test_df.reset_index(inplace=True)
    test_df[out_cols].to_csv(f'{path}/fc_pred.csv', index=False)
