import pandas as pd
import numpy as np
from fastai.structured import apply_cats
from sklearn.metrics import mean_squared_error


def transform_columns(df, cat_vars, cont_vars):
    for v in cat_vars:
        df[v] = df[v].astype('category').cat.as_ordered()
    for v in cont_vars:
        df[v] = df[v].fillna(0).astype('float32')


def get_embedding_sizes(cat_vars, df):
    cat_sz = [(c, len(df[c].cat.categories) + 1) for c in cat_vars]
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


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

