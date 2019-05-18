import pandas as pd

from stock.train import add_put_call_features, add_ticker_features
from common.lgb import LGBModel

def main():
    path = 'data'
    df = pd.read_csv(f'{path}/spy.csv')
    add_ticker_features(df, 'spy')
    df = add_put_call_features(df, path)
    df.dropna(inplace=True)

    feat_cols = []
    for c in df.columns:
        if c.startswith('spy_d') or c.startswith('pc_'):
            feat_cols.append(c)

    train_end = int(len(df) * 3 / 5)
    model = LGBModel('pc_spy_lgb', path, 'timestamp', 'target', 5, feat_cols)
    model.predict(df.iloc[train_end:])

    print('done')

if __name__ == '__main__':
    main()