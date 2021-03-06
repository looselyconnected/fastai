import numpy as np
import pandas as pd

import sklearn.datasets

from common.data import get_embedding_sizes, get_validation_index

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from fastai.structured import *
from fastai.metrics import *
from fastai.column_data import ColumnarModelData, StructuredLearner, StructuredModel
from fastai.sgdr import LossRecorder
from fastai.core import to_gpu

import torch.nn as nn

from common.fc import roc_auc, metrics_map, SaveBestModel

import torch.nn.functional as F


class ClassifierModel(nn.Module):
    def __init__(self, input_size, output_size, conv_features=[16]):
        super().__init__()
        # self.convs = []
        # prev_f_size = 1
        # for f_size in conv_features:
        #     self.convs += [nn.Conv1d(prev_f_size, f_size, 1).cuda()]
        #     prev_f_size = f_size
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv5 = nn.Conv1d(32, 16, 3, padding=1)
        self.conv6 = nn.Conv1d(16, 16, 3, padding=1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)
        self.dropout5 = nn.Dropout(p=0.1)
        self.dropout6 = nn.Dropout(p=0.1)
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.MaxPool1d(2, padding=1)
        self.pool4 = nn.MaxPool1d(2, padding=1)
        self.pool5 = nn.MaxPool1d(2, padding=1)
        self.pool6 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 1)
        self.output_size = output_size

    def forward(self, x):
        # for conv in self.convs:
        #     x = F.relu(conv(x))
        x = F.relu(self.dropout1(self.pool1(self.conv1(x))))
        x = F.relu(self.dropout2(self.pool2(self.conv2(x))))
        x = F.relu(self.dropout3(self.pool3(self.conv3(x))))
        x = F.relu(self.dropout4(self.pool4(self.conv4(x))))
        x = F.relu(self.dropout5(self.pool5(self.conv5(x))))
        x = F.relu(self.dropout6(self.pool6(self.conv6(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))

        if self.output_size == 1:
            x = torch.sigmoid(self.fc2(x)).view(-1)
        else:
            x = F.softmax(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Given a model, x, y, and index get the loss
def train_batch(model, criterion, x, y):
    batch_pred = model(torch.Tensor(x).cuda())
    loss = criterion(batch_pred, torch.Tensor(y).cuda())
    return batch_pred, loss


# 1-d conv net
def kfold_cnn(train_df, test_df, num_folds, params, path, label_col, target_col,
             feats_excluded=None, out_cols=None, stratified=False, cat_cols=[], name=None, static=False):
    print("Starting CNN. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

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

    train_x = np.reshape(train_x.values, (len(train_x), 1, len(feat_cols)))
    test_x = np.reshape(test_x.values, (len(test_x), 1, len(feat_cols)))

    train_metrics = None
    param_metrics = params.get('metrics')
    if param_metrics is not None:
        train_metrics = []
        for metric in param_metrics:
            train_metrics.append(metrics_map[metric])

    criterion = nn.MSELoss().cuda()
    epochs = params.get('epochs', 10)
    batch_size = 128

    if static:
        num_folds = 1
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df[feat_cols], train_df[target_col])):
        best_metrics = 0
        best_loss = 1e10
        best_epoch = 0
        print("Fold {}".format(fold))
        model_name = f'{name}-{fold}'
        model = ClassifierModel(train_x.shape[2], 1, conv_features=params.get('layers'))
        model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=params.get('lr', 1e-3), momentum=0.9)
        try:
            model.load_state_dict(torch.load(f'{path}/models/{model_name}'))
            print(f'loaded model from {path}/models/{model_name}')
        except FileNotFoundError:
            pass

        for epoch in range(epochs):
            # go through all the training samples in batches
            current = 0
            while current < len(train_idx):
                end = min(current + batch_size, len(train_idx))
                batch_idx = train_idx[current:end]
                current = end

                _, loss = train_batch(model, criterion, train_x[batch_idx], train_y[batch_idx])
                # if (current / batch_size) % 100 == 0:
                #     print(f'epoch {epoch}, {current/batch_size}, loss {loss.item()}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            valid_y = train_y[valid_idx]
            pred_y, loss = train_batch(model, criterion, train_x[valid_idx], valid_y)
            metric = 0
            if len(train_metrics) > 0:
                metric = train_metrics[0](pred_y.cpu().detach().numpy(), valid_y)
            print(f'epoch {epoch} done, loss {loss.item()}, metrics: {metric}')

            if metric > best_metrics or (metric == best_metrics and loss < best_loss):
                best_epoch = epoch
                best_metrics = metric
                torch.save(model.state_dict(), f'{path}/models/{model_name}')
                print('best')

            if epoch > best_epoch + params.get('early_stopping', 0):
                print(f'stopping, best epoch is {best_epoch}')
                break

        # load the best model
        print(f'Best epoch is {best_epoch} loss {best_loss} metric {best_metrics}')
        model.load_state_dict(torch.load(f'{path}/models/{model_name}'))
        model.eval()
        current = 0
        while current < len(test_df):
            end = min(current + 10000, len(test_df))
            test_y = model(torch.Tensor(test_x[current:end]).cuda()).cpu().detach().numpy()
            test_df.loc[current:end-1, target_col] += (test_y / num_folds)
            current = end

        if static:
            break

    # save submission file
    test_df.reset_index(inplace=True)
    test_df[out_cols].to_csv(f'{path}/cnn_pred.csv', index=False)

