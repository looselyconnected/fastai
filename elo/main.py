import numpy as np

from fastai.structured import *
from fastai.column_data import ColumnarModelData

import matplotlib.pyplot as plt

np.set_printoptions(threshold=50, edgeitems=20)

PATH = 'experiments/'

def main():
    tables = [pd.read_csv(f'{PATH}/{fname}.csv', low_memory=False)
              for fname in ['train', 'test', 'new_merchant_transactions', 'merchants', 'historical_transactions']]
    train, test, new_trans, merchants, trans = tables
    add_datepart(trans, 'purchase_date', drop=True)
    add_datepart(new_trans, 'purchase_date', drop=True)

    # TODO in trans we should add days since last purchase at the merchant, use -1 as the first time

    # First, train card_id embedding using the trans, new_trans and merchants data.
    df = pd.merge(trans, merchants, on=['merchant_id', 'merchant_category_id', 'subsector_id', 'city_id', 'state_id',
                                        'category_1', 'category_2'], how='left')
    df = df[~df.avg_sales_lag12.isnull()]

    # need to add date specific fields for purchase_date
    cat_vars = ['authorized_flag', 'card_id', 'city_id', 'category_1', 'installments',
       'category_3', 'merchant_category_id', 'merchant_id', 'month_lag',
       'purchase_date', 'category_2', 'state_id',
       'subsector_id', 'merchant_group_id',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'active_months_lag3', 'active_months_lag6', 'active_months_lag12',
       'category_4']
    num_vars = ['numerical_1', 'numerical_2', 'avg_sales_lag3', 'avg_purchases_lag3',
                'avg_sales_lag6', 'avg_purchases_lag6', 'avg_sales_lag12', 'avg_purchases_lag12']

    print('done')


if __name__ == "__main__":
    main()
