import datetime
import gc
import numpy as np
import pandas as pd
import warnings

from pandas.core.common import SettingWithCopyWarning

from common.data import *
from common.lgb import lgb_train
from elo.embedding import train_card_merchant_embeddings, load_card_merchant_pct
from elo.word2vec import load_word2vec_embeddings

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']

PATH = 'experiments/'

# preprocessing train & test
def train_test(num_rows=None):
    # load csv
    train_df = pd.read_csv(f'{PATH}/train.csv', index_col=['card_id'], nrows=num_rows)
    test_df = pd.read_csv(f'{PATH}/test.csv', index_col=['card_id'], nrows=num_rows)

    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    # outlier
    train_df['outliers'] = 0
    train_df.loc[train_df['target'] < -30, 'outliers'] = 1

    # set target as nan
    test_df['target'] = np.nan

    # merge
    df = train_df.append(test_df)

    del train_df, test_df
    gc.collect()

    # to datetime
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    # datetime features
    df['quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['days_feature1'] = df['elapsed_time'] * df['feature_1']
    df['days_feature2'] = df['elapsed_time'] * df['feature_2']
    df['days_feature3'] = df['elapsed_time'] * df['feature_3']

    df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
    df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
    df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

    # one hot encoding
    df, cols = one_hot_encoder(df, nan_as_category=False)

    for f in ['feature_1', 'feature_2', 'feature_3']:
        order_label = df.groupby([f])['outliers'].mean()
        df[f] = df[f].map(order_label)

    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum'] / 3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    return df


# preprocessing historical transactions
def add_transaction_features(hist_df, col_prefix, num_rows=None):
    # fillna
    hist_df['category_2'].fillna(1.0, inplace=True)
    hist_df['category_3'].fillna('A', inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    hist_df['installments'].replace(-1, np.nan, inplace=True)
    hist_df['installments'].replace(999, np.nan, inplace=True)

    # trim
    # hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))
    hist_df.loc[hist_df['purchase_amount'] > 0.8, 'purchase_amount'] = 0.8

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A': 0, 'B': 1, 'C': 2})

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >= 5).astype(int)

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

    # XXX This should be same as without introducing today
    hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days) // 30
    hist_df['month_diff'] += hist_df['month_lag']

    # additional features
    hist_df['duration'] = hist_df['purchase_amount'] * hist_df['month_diff']
    hist_df['amount_month_ratio'] = hist_df['purchase_amount'] / hist_df['month_diff']

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var', 'skew']
    aggs['installments'] = ['sum', 'max', 'mean', 'var', 'skew']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['month_diff'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['authorized_flag'] = ['mean']
    aggs['weekend'] = ['mean']  # overwrite
    aggs['weekday'] = ['mean']  # overwrite
    aggs['day'] = ['nunique', 'mean', 'min']  # overwrite
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['sum', 'mean', 'max', 'min', 'var']
    aggs['duration'] = ['mean', 'min', 'max', 'var', 'skew']
    aggs['amount_month_ratio'] = ['mean', 'min', 'max', 'var', 'skew']

    for col in ['category_2', 'category_3']:
        hist_df[col + '_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        hist_df[col + '_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
        hist_df[col + '_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
        hist_df[col + '_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']

    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])

    hist_df['purchase_date_diff'] = (hist_df['purchase_date_max'] - hist_df['purchase_date_min']).dt.days
    hist_df['purchase_date_average'] = hist_df['purchase_date_diff'] / hist_df['card_id_size']
    hist_df['purchase_date_uptonow'] = (datetime.datetime.today() - hist_df['purchase_date_max']).dt.days
    hist_df['purchase_date_uptomin'] = (datetime.datetime.today() - hist_df['purchase_date_min']).dt.days

    # add prefix
    hist_df.columns = [col_prefix + c for c in hist_df.columns]

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    return hist_df


# additional features
def additional_features(df):
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
    df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

    date_features = ['hist_purchase_date_max', 'hist_purchase_date_min',
                     'new_purchase_date_max', 'new_purchase_date_min']

    for f in date_features:
        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_card_id_size'] + df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_card_id_count'] + df['hist_card_id_count']
    df['card_id_cnt_ratio'] = df['new_card_id_count'] / df['hist_card_id_count']
    df['purchase_amount_total'] = df['new_purchase_amount_sum'] + df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean'] + df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max'] + df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min'] + df['hist_purchase_amount_min']
    df['purchase_amount_ratio'] = df['new_purchase_amount_sum'] / df['hist_purchase_amount_sum']
    df['month_diff_mean'] = df['new_month_diff_mean'] + df['hist_month_diff_mean']
    df['month_diff_ratio'] = df['new_month_diff_mean'] / df['hist_month_diff_mean']
    df['month_lag_mean'] = df['new_month_lag_mean'] + df['hist_month_lag_mean']
    df['month_lag_max'] = df['new_month_lag_max'] + df['hist_month_lag_max']
    df['month_lag_min'] = df['new_month_lag_min'] + df['hist_month_lag_min']
    df['category_1_mean'] = df['new_category_1_mean'] + df['hist_category_1_mean']
    df['installments_total'] = df['new_installments_sum'] + df['hist_installments_sum']
    df['installments_mean'] = df['new_installments_mean'] + df['hist_installments_mean']
    df['installments_max'] = df['new_installments_max'] + df['hist_installments_max']
    df['installments_ratio'] = df['new_installments_sum'] / df['hist_installments_sum']
    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']
    df['duration_mean'] = df['new_duration_mean'] + df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min'] + df['hist_duration_min']
    df['duration_max'] = df['new_duration_max'] + df['hist_duration_max']
    df['amount_month_ratio_mean'] = df['new_amount_month_ratio_mean'] + df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min'] = df['new_amount_month_ratio_min'] + df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max'] = df['new_amount_month_ratio_max'] + df['hist_amount_month_ratio_max']
    df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

    return df


def load_card_merchant_embeddings():
    c_m_cat = load_file(f'{PATH}/card_merchant_pct_cat.hdf')
    learner = train_card_merchant_embeddings(c_m_cat, PATH, 0)
    card_cat_df = pd.DataFrame(c_m_cat.card_id.cat.categories, columns=['card_id'])
    card_emb_df = pd.DataFrame(learner.model.embs[0].weight.data.numpy())
    card_emb_df['card_id'] = card_cat_df['card_id']
    return card_emb_df


def load_word2vec_merchant_embeddings():
    emb_df = load_word2vec_embeddings(PATH)
    c_m_pct = load_card_merchant_pct(PATH, debug=False)
    c_m_pct = c_m_pct.merge(emb_df, on=['merchant_id'], how='left')
    for col in emb_df.columns:
        c_m_pct[col] = c_m_pct[col] * c_m_pct.percent

    return c_m_pct.groupby('card_id').sum()


def get_train_test_with_features(debug=False):
    try:
        df = pd.read_hdf(f'{PATH}hdf_lgb', 'train_test')
    except Exception:
        df = None

    if df is None:
        num_rows = 10000 if debug else None
        with timer("train & test"):
            df = train_test(num_rows)
        with timer("historical transactions"):
            # load csv
            trans = pd.read_csv(f'{PATH}/historical_transactions.csv', nrows=num_rows)
            df = pd.merge(df, add_transaction_features(trans, 'hist_', num_rows), on='card_id', how='outer')
        with timer("new merchants"):
            trans = pd.read_csv(f'{PATH}/new_merchant_transactions.csv', nrows=num_rows)
            df = pd.merge(df, add_transaction_features(trans, 'new_', num_rows), on='card_id', how='outer')
        # with timer("additional features"):
        #     df = additional_features(df)

        if not debug:
            df.to_hdf(f'{PATH}hdf_lgb', 'train_test')
    return df


def lgb_run(debug=False):
    df = get_train_test_with_features(debug)

    # load the embeddings
    # card_emb_df = load_card_merchant_embeddings()
    card_emb_df = load_word2vec_merchant_embeddings()
    df = df.merge(card_emb_df, on=['card_id'], how='left')

    with timer("split train & test"):
        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]
        del df
        gc.collect()
    with timer("Run LightGBM with kfold"):
        # params optimized by optuna
        params = {
            'task': 'train',
            'boosting': 'goss',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9855232997390695,
            'max_depth': 7,
            'top_rate': 0.9064148448434349,
            'num_leaves': 63,
            'min_child_weight': 41.9612869171337,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 9.677537745007898,
            'colsample_bytree': 0.5665320670155495,
            'min_split_gain': 9.820197773625843,
            'reg_lambda': 8.2532317400459,
            'min_data_in_leaf': 21,
            'verbose': -1,
        }

        lgb_train(train_df, num_folds=5, feats_excluded=FEATS_EXCLUDED, path=PATH, params=params,
                       label_col='card_id', target_col='target', stratified=False, debug=debug)

    print('done')
