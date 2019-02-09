import pandas as pd
import gc
import numpy as np

from fastai.structured import proc_df
from fastai.column_data import ColumnarModelData

from common.data import load_file, save_file, set_common_categorical, get_validation_index, get_embedding_sizes


def prepare_card_merchant_pct(path, debug):
    num_rows = 10000 if debug else None

    hist = pd.read_csv(f'{path}/historical_transactions.csv', nrows=num_rows, usecols=['card_id', 'merchant_id'])
    new = pd.read_csv(f'{path}/historical_transactions.csv', nrows=num_rows, usecols=['card_id', 'merchant_id'])

    hist = hist[hist.merchant_id.notna()]
    new = new[new.merchant_id.notna()]

    df = pd.concat([hist, new])
    df['count'] = 1

    c_m_count_df = df.groupby(['card_id', 'merchant_id']).count().reset_index()
    c_count_df = c_m_count_df.groupby(['card_id']).sum()

    merged_df = c_m_count_df.merge(c_count_df[['count']], on=['card_id'], how='left')
    merged_df['percent'] = merged_df['count_x'] / merged_df['count_y']
    merged_df.drop(['count_x', 'count_y'], inplace=True, axis=1)

    # Add some random negative samples. Because these doesn't exist, the count is set to 0 initially
    neg_df = merged_df.copy()
    neg_df['percent'] = 0
    neg_df['merchant_id'] = hist.loc[hist.sample(n=len(neg_df)).index]['merchant_id'].reset_index(drop=True)

    all_df = pd.concat([merged_df, neg_df])

    # Because there may be redundant entries, must sum again
    all_df = all_df.groupby(['card_id', 'merchant_id']).sum().reset_index()
    save_file(all_df, f'{path}/card_merchant_pct.hdf')
    return all_df


def load_card_merchant_pct(path, debug):
    df = load_file(f'{path}/card_merchant_pct.hdf')
    if df is None or len(df) == 0:
        df = prepare_card_merchant_pct(path, debug)

    return df


def prepare_train_test_category(path, card_merchant_pct, debug):
    num_rows = 10000 if debug else None
    train = pd.read_csv(f'{path}/train.csv', nrows=num_rows)
    test = pd.read_csv(f'{path}/test.csv', nrows=num_rows)

    set_common_categorical([card_merchant_pct, train, test], 'card_id')
    save_file(card_merchant_pct, f'{path}/card_merchant_pct_cat.hdf')
    save_file(train, f'{path}/train_cat.hdf')
    save_file(test, f'{path}/test_cat.hdf')

    return train, test


def load_all_category(path, debug):
    train = load_file(f'{path}/train_cat.hdf')
    test = load_file(f'{path}/test_cat.hdf')
    c_m_cat = load_file(f'{path}/card_merchant_pct_cat.hdf')
    if train is None or test is None or c_m_cat is None:
        c_m_cat = load_card_merchant_pct(path, debug)
        train, test = prepare_train_test_category(path, c_m_cat, debug)

    return c_m_cat, train, test


def train_card_merchant_embeddings(df, path, iter=1):
    df['merchant_id'] = df['merchant_id'].astype('category').cat.as_ordered()
    val_idx = get_validation_index(df, frac=0.2, random=True)
    x, y, nas = proc_df(df, 'percent', do_scale=False)

    cat_vars = ['card_id', 'merchant_id']
    md = ColumnarModelData.from_data_frame(path, val_idx, x, y.astype(np.float32), cat_flds=cat_vars,
                                           is_reg=True, is_multi=False, bs=128, test_df=None)
    embedding_sizes = get_embedding_sizes(cat_vars, df)
    learner = md.get_learner(embedding_sizes, 0, 0.1, 1, [20, 5], [0.1, 0.1],
                             y_range=(0.0, 1.0))

    # Load model, train, save
    try:
        learner.load('embedding_model')
    except FileNotFoundError:
        pass

    for i in range(iter):
        print(f'training iter {i}')
        learner.fit(1e-2, 1)
        learner.save('embedding_model')

    return learner

# Train the user and merchant embeddings using per user merchant purchase percentage as target.
def train_embeddings(path, debug=False):
    c_m_df, train, test = load_all_category(path, debug)

    del train, test
    gc.collect()

    train_card_merchant_embeddings(c_m_df, path, iter=5)
    return