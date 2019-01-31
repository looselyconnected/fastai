import numpy as np

from os.path import isfile
from scipy.stats import describe
from torch.nn.parameter import Parameter

from fastai.structured import *
from fastai.column_data import ColumnarModelData

import matplotlib.pyplot as plt

from common.data import *
from elo.lgb import lgb_run

np.set_printoptions(threshold=50, edgeitems=20)

PATH = 'experiments/'
MODEL = 'embedding_model'
cat_vars = ['card_id', 'authorized_flag', 'city_id', 'category_1', 'installments',
            'category_3', 'merchant_category_id', 'merchant_id', 'month_lag',
            'category_2', 'state_id',
            'subsector_id', 'merchant_group_id',
            'most_recent_sales_range', 'most_recent_purchases_range',
            'active_months_lag3', 'active_months_lag6', 'active_months_lag12',
            'category_4', 'purchase_Month', 'purchase_Week', 'purchase_Day',
            'purchase_Dayofweek']
cont_vars = ['numerical_1', 'numerical_2', 'avg_sales_lag3', 'avg_purchases_lag3',
             'avg_sales_lag6', 'avg_purchases_lag6', 'avg_sales_lag12', 'avg_purchases_lag12']


def get_card_stats(df, card_id):
    trans_count = len(df)
    days = (df.iloc[len(df)-1].purchase_Elapsed - df.iloc[0].purchase_Elapsed) / 86400
    merchant_count = len(df.merchant_id.unique())
    sector_count = len(df.subsector_id.unique())
    num_vars = ['numerical_1', 'numerical_2', 'avg_sales_lag3',
                'avg_purchases_lag3', 'avg_sales_lag6', 'avg_purchases_lag6',
                'avg_sales_lag12', 'avg_purchases_lag12']
    mean_rc = dict(df[num_vars].mean())
    rc = {'card_id': card_id, 'trans_per_day': trans_count / days,
          'trans_per_merchant_day': trans_count / days / merchant_count,
          'merchants_per_sector': merchant_count / sector_count}
    return {**mean_rc, **rc}


def prepare_data():
    tables = [pd.read_csv(f'{PATH}/{fname}.csv', low_memory=False)
              for fname in ['train', 'test', 'new_merchant_transactions', 'merchants', 'historical_transactions']]
    train, test, new_trans, merchants, trans = tables
    add_datepart(trans, 'purchase_date', drop=False)
    add_datepart(new_trans, 'purchase_date', drop=False)

    all_trans = pd.concat([trans, new_trans]).set_index(['purchase_date'])
    all_trans.drop(columns=['merchant_category_id', 'subsector_id', 'city_id',
                            'state_id', 'category_1', 'category_2'], inplace=True)
    merchants = merchants.groupby('merchant_id').first()

    df = pd.merge(all_trans, merchants, on=['merchant_id'], how='left')

    # Go through all the transactions and generate the aggregate card_id features
    df.sort_values(by=['card_id', 'purchase_Elapsed'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    card_list = []
    start_index = 0
    current_id = df.iloc[0].card_id
    for index, row in df.iterrows():
        if row.card_id != current_id:
            card_df = df.iloc[start_index : index]
            card_list.append(get_card_stats(card_df, current_id))

            start_index = index
            current_id = row.card_id

            if len(card_list) % 100 == 0:
                print(f'{len(card_list)} cards processed for features')

    card_df = df.iloc[start_index: len(df)-1]
    card_list.append(get_card_stats(card_df, current_id))

    card_features = pd.DataFrame(card_list)
    train = train.merge(card_features, on=['card_id'], how='left')
    test = test.merge(card_features, on=['card_id'], how='left')

    # save all
    df.sort_values(by=['purchase_Elapsed'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_feather(f'{PATH}joined')
    train.to_feather(f'{PATH}train')
    test.to_feather(f'{PATH}test')

    return df, train, test


def load_data():
    joined = load_file(f'{PATH}/joined')
    if joined is None:
        return None, None, None
    train = load_file(f'{PATH}/train')
    if train is None:
        return None, None, None
    test = load_file(f'{PATH}/test')
    if test is None:
        return None, None, None
    return joined, train, test


# Train card embedding. If iter is 1 just return the existing saved model
def train_card_embeddings(df, iter=1):
    # We may use a smaller set of data to get a sense of the performance of the model, comment out before final
    # training
    # df = df.sample(frac=0.1)

    df = df[cat_vars + cont_vars + ['purchase_amount']]
    df = df[(df.purchase_amount < 5) & (df.avg_purchases_lag12 < 5) & (df.avg_sales_lag12 < 5)]
    df.reset_index(inplace=True, drop=True)
    val_idx = get_validation_index(df, frac=0.25, random=False)

    # make cat_vars, but card_id needs special treatment
    cat_var_no_cid = cat_vars.copy()
    cat_var_no_cid.remove('card_id')
    transform_columns(df, cat_var_no_cid, cont_vars)

    x, y, nas, mapper = proc_df(df, 'purchase_amount', do_scale=True)

    md = ColumnarModelData.from_data_frame(PATH, val_idx, x, y.astype(np.float32), cat_flds=cat_vars,
                                           is_reg=True, is_multi=False, bs=128, test_df=None)
    embedding_sizes = get_embedding_sizes(cat_vars, df)
    learner = md.get_learner(embedding_sizes, len(x.columns) - len(cat_vars), 0.5, 1, [50, 7], [0.5, 0.5],
                             y_range=(-1.0, 0.0))


    # Load model, train, save
    try:
        learner.load(MODEL)
    except FileNotFoundError:
        pass

    for i in range(iter):
        print(f'training iter {i}')
        learner.fit(1e-2, 1)
        learner.save(MODEL)

    return learner


def predict_and_save(learner, test, fname):
    pred_test = learner.predict(True)
    tc = test.copy()
    tc.loc[:, 'target'] = pred_test
    tc[['card_id', 'target']].to_csv(f'{PATH}/tmp/{fname}.csv', index=False)


def train_full():
    df, train, test = load_data()
    if df is None or train is None or test is None:
        df, train, test = prepare_data()

    set_common_categorical([df, train, test], 'card_id')

    card_learner = train_card_embeddings(df, 1)

    # Get the card_id embedding out

    # training and testing with the real train/test set
    train_cat_flds = ['card_id', 'first_active_month']
    set_common_categorical([train, test], 'first_active_month')
    test['target'] = 0

    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    train.reset_index(inplace=True, drop=True)
    train_x, train_y, nas, mapper = proc_df(train, 'target', do_scale=True)
    test_x, _, nas, mapper = proc_df(test, 'target', do_scale=True, mapper=mapper, na_dict=nas)
    train_val_idx = get_validation_index(train, frac=0.25)
    md = ColumnarModelData.from_data_frame(PATH, train_val_idx, train_x, train_y.astype(np.float32),
                                           cat_flds=train_cat_flds, is_reg=True, bs=128, test_df=test_x)
    embedding_sizes = get_embedding_sizes(train_cat_flds, train)
    learner = md.get_learner(embedding_sizes, len(train_x.columns) - len(train_cat_flds), 0.5, 1, [20, 5], [0.5, 0.5],
                             y_range=(-35.0, 20.0))

    # learner.lr_find()
    # learner.sched.plot(100)
    learner.model.embs[0].weight = Parameter(card_learner.model.embs[0].weight.data.clone())
    learner.model.embs[0].weight.requires_grad = False

    learner.fit(1e-3, 10)
    predict_and_save(learner, test, 'base')

    print('done')


def train_no_card_embedding():
    train = load_file('train')
    test = load_file('test')

    # training and testing with the real train/test set
    train_cat_flds = ['first_active_month']
    set_common_categorical([train, test], 'first_active_month')
    test['target'] = 0

    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    train.reset_index(inplace=True, drop=True)
    train_x, train_y, nas, mapper = proc_df(train, 'target', do_scale=True, skip_flds=['card_id'])
    test_x, _, nas, mapper = proc_df(test, 'target', do_scale=True, mapper=mapper, na_dict=nas, skip_flds=['card_id'])
    train_val_idx = get_validation_index(train, frac=0.25)
    md = ColumnarModelData.from_data_frame(PATH, train_val_idx, train_x, train_y.astype(np.float32),
                                           cat_flds=train_cat_flds, is_reg=True, bs=128, test_df=test_x)
    embedding_sizes = get_embedding_sizes(train_cat_flds, train)
    learner = md.get_learner(embedding_sizes, len(train_x.columns) - len(train_cat_flds), 0.5, 1, [20, 5], [0.5, 0.5],
                             y_range=(-35.0, 20.0))

    # learner.lr_find()
    # learner.sched.plot(100)
    try:
        learner.load('no_card_embedding')
    except FileNotFoundError:
        pass

    for i in range(10):
        learner.fit(1e-3, 20)
        learner.save(f'no_card_embedding_{i}')

        predict_and_save(learner, test, f'base_{i}')

    print('done')


def main():
    train_no_card_embedding()

if __name__ == "__main__":
    # main()
    lgb_run(debug=False)
