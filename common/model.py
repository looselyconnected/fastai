import abc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from common.data import prediction_to_df


class MLSplit:
    def __init__(self, random_shuffle, train_percent):
        self.random_shuffle = random_shuffle
        self.train_percent = train_percent

    def split(self, train_x, y=None, groups=None):
        train_len = len(train_x)
        train_percent = self.train_percent
        if self.random_shuffle:
            train_mask = np.random.choice([True, False], train_len, p=[train_percent, 1.0 - train_percent])
        else:
            train_mask = np.append([True] * int(train_percent * train_len),
                                   [False] * (train_len - int(train_percent * train_len)))
        train_idx = train_mask
        valid_idx = ~train_mask
        yield train_idx, valid_idx


class MLModel(abc.ABC):
    def __init__(self, name, path, label_col, target_col, num_folds=0, feat_cols=None, out_cols=None):
        self.model = None
        self.name = name
        self.path = path
        self.fold_model_path = None
        self.label_col = label_col
        self.target_col = target_col
        self.num_folds = num_folds
        self.feat_cols = feat_cols
        self.out_cols = out_cols
        print(f'features {self.feat_cols}')

    @abc.abstractclassmethod
    def load(self, model_path):
        assert False

    @abc.abstractclassmethod
    def save(self, model_path):
        assert False

    def pre_fold(self, fold):
        model_name = f'{self.name}-{fold}'
        self.fold_model_path = f'{self.path}/models/{model_name}'
        try:
            self.load(self.fold_model_path)
            print(f'loaded model from {self.fold_model_path}')
        except:
            pass

    def post_fold(self, fold):
        self.save(self.fold_model_path)

    @abc.abstractclassmethod
    def train_one_fold(self, fold, params, train_df, train_idx, valid_idx):
        assert False

    @abc.abstractclassmethod
    def predict_one_fold(self, df):
        assert False

    def pre_train(self):
        return

    def post_train(self):
        return

    def train(self, train_df, params, stratified=False, random_shuffle=True):
        print("Starting training. Train shape: {}".format(train_df.shape))

        # Cross validation model
        if self.num_folds > 1:
            if stratified:
                kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=326)
            else:
                kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=326)
        else:
            kf = MLSplit(random_shuffle, 0.8)

        self.pre_train()
        # k-fold
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df[self.feat_cols], train_df[self.target_col])):
            print("Fold {}".format(fold + 1))

            self.pre_fold(fold)
            self.train_one_fold(fold, params, train_df, train_idx, valid_idx)
            self.post_fold(fold)

        self.post_train()

    def predict(self, df):
        include_header = False
        pred_file = f'{self.path}/{self.name}_pred.csv'
        try:
            last_pred_time = pd.read_csv(pred_file).iloc[-1].timestamp
            df = df[df.timestamp > last_pred_time].copy()
            if len(df) == 0:
                return
        except:
            include_header = True

        sub_preds = None
        for fold in range(self.num_folds):
            self.pre_fold(fold)
            pred = self.predict_one_fold(df[self.feat_cols]) / self.num_folds
            if sub_preds is None:
                sub_preds = np.zeros(pred.shape)
            sub_preds += pred

        pred_df = prediction_to_df(self.target_col, sub_preds)
        df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

        if self.out_cols is None:
            self.out_cols = [self.label_col] + pred_df.columns.tolist()

        if include_header:
            df[self.out_cols].to_csv(pred_file, index=False)
        else:
            out_csv = df[self.out_cols].to_csv(index=False, header=include_header)
            f = open(pred_file, 'a')
            f.write(out_csv)
