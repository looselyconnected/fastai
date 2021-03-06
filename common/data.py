import pandas as pd
import numpy as np
from fastai.structured import apply_cats
from io import StringIO
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from contextlib import contextmanager
from sklearn.metrics import mean_squared_error


def transform_columns(df, cat_vars, cont_vars):
    for v in cat_vars:
        df[v] = df[v].astype('category').cat.as_ordered()
    for v in cont_vars:
        df[v] = df[v].fillna(0).astype('float32')


def get_embedding_sizes(cat_vars, df):
    cat_sz = [(c, len(df[c].cat.categories)) for c in cat_vars]
    embedding_sizes = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
    return embedding_sizes


def get_validation_index(df, frac=0.25, random=True):
    if random:
        return df.sample(frac=frac).index
    else:
        total = len(df)
        return list(range(int(total - total*frac), total))


def lr_find(learner, start_lr=1e-4, end_lr=1):
    learner.lr_find(start_lr=start_lr, end_lr=end_lr)
    learner.sched.plot(100)


# the dfs is a list of dataframes, the cols is a list of corresponding column names to be set as
# categorical. This function makes sure that the categorical var of the columns maps to the same hash table.
def set_common_categorical(dfs, col):
    all_df = pd.DataFrame([], columns=[col])
    for df in dfs:
        all_df = pd.concat([all_df, df[[col]]])

    all_df[col] = all_df[col].astype('category').cat.as_ordered()
    for df in dfs:
        apply_cats(df, all_df)


def load_file(fname):
    try:
        df = pd.read_hdf(fname, 'k')
        return df
    except Exception:
        return None


def save_file(df, fname):
    df.to_hdf(fname, 'k', format='table')


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, prefix=None, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, prefix=prefix, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Display/plot feature importance
def display_importances(feature_importance_df_):
    sorted_df = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)
    print(sorted_df)

    threshold = 40
    cols = sorted_df[:threshold].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


# reduce memory
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def set_to_float32(df):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float64']
    for col in df.columns:
        if df[col].dtypes in numerics:
            df[col] = df[col].fillna(0.0).astype('float32')


def add_stat_features(df, cols):
    for feature in cols:
        df['mean_' + feature] = (df[feature] - df[feature].mean())
        df['z_' + feature] = df['mean_' + feature] / df[feature].std(ddof=0)
        df['sq_' + feature] = (df[feature]) ** 2
        df['sqrt_' + feature] = np.abs(df[feature]) ** (1 / 2)
        df['log_' + feature] = np.log(df['sq_' + feature] + 10) / 2


def prediction_to_df(target_col, pred):
    if len(pred.shape) == 2:
        pred_cols = [f'{target_col}_{i}' for i in range(pred.shape[1])]
    else:
        pred_cols = [target_col]
    return pd.DataFrame(pred, columns=pred_cols)


# Given a csv file, return a dataframe with just the last row
def get_last_row(filename):
    try:
        statinfo = os.stat(filename)
        f = open(filename)
        header = f.readline()
        f.seek(0 if statinfo.st_size <= 4096 else statinfo.st_size - 4096, 0)
        last_rows = f.read(4096).split('\n')
        last_row = last_rows[-1]
        if len(last_row) == 0:
            last_row = last_rows[-2]
        str = header + last_row
        return pd.read_csv(StringIO(str))
    except:
        return None


# Append the diffs to the end of the file
def append_diff_to_csv(filename, df, fieldname):
    orig = pd.read_csv(filename)
    combined = pd.merge(orig, df, on=df.columns.tolist(), how='outer')
    diff = combined.loc[~combined[fieldname].isin(orig[fieldname])]
    if len(diff) == 0:
        return
    diff_csv = diff.to_csv(header=False, index=False)
    f = open(filename, 'a')
    f.write(diff_csv)

