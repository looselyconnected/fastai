
import pandas as pd
from stock.test import *

def main():
    portfolio = Portfolio(1, 'data', 'index_by_size.csv')
    pred = pd.read_csv('data/pc_spy_lgb_pred.csv')
    res = []
    for _, row in pred.iterrows():
        if row.target > 0.5:
            holdings = ['spy']
        else:
            holdings = ['cash']
        portfolio.set_desired(row.timestamp, holdings)
        res += [holdings + [portfolio.value]]

    portfolio.liquidate(pred.iloc[-1].timestamp)

    print(f'\nfrom {pred.iloc[0].timestamp} to {pred.iloc[-1].timestamp} holding gain {portfolio.cash}')
    print(portfolio.histogram)

    plt.plot(pd.DataFrame(res)[1])

    const_df = test_holding_constant('data', f'index_by_size.csv', 'spy', pred.iloc[0].timestamp)
    plt.plot(const_df.adjusted_close / const_df.iloc[0].adjusted_close)

    plt.savefig('test_compare.png')

    return

if __name__ == '__main__':
    main()
