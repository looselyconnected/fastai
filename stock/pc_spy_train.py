import pandas as pd

from stock.train import add_put_call_features, add_ticker_features
from common.lgb import LGBModel

def main():
    path = 'data'
    df = pd.read_csv(f'{path}/spy.csv')
    add_ticker_features(df, 'spy')
    df = add_put_call_features(df, path)
    df_copy = df.copy()
    df_copy.index -= 5
    df['target'] = df_copy.adjusted_close - df.adjusted_close
    df.dropna(inplace=True)
    df.target = df.target > 0

    feat_cols = []
    for c in df.columns:
        if c.startswith('spy_d') or c.startswith('pc_'):
            feat_cols.append(c)

    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.005,
        'verbose': 1,

        'num_rounds': 30000,
        #'is_unbalance': True,
        #'scale_pos_weight': 8.951238929246692,
        'early_stopping': 1000,

        # 'bagging_freq': 5,
        # 'bagging_fraction': 0.33,
        'boost_from_average': 'false',
        'feature_fraction': 1.0,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        # 'min_sum_hessian_in_leaf': 5.0,
        'num_leaves': 3,
        'num_threads': 8,
        'tree_learner': 'serial',
    }

    train_end = int(len(df) * 3 / 5)
    model = LGBModel('pc_spy_lgb', path, 'timestamp', 'target', 5, feat_cols)
    model.train(df.iloc[0:train_end], params, stratified=False, random_shuffle=True)

    print('done')

if __name__ == '__main__':
    main()