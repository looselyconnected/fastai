import pandas as pd
import numpy as np
from stock.data import Fields as fld, get_ticker_df, index_to_map


class Portfolio:
    def __init__(self, max_holdings, ticker_dfs):
        self.holdings = []
        self.cash = 1.0
        self.max_holdings = max_holdings
        self.ticker_dfs = ticker_dfs
        for _, v in self.ticker_dfs.items():
            v.set_index('timestamp', inplace=True)

    def get_ticker_price(self, timestamp, ticker):
        if ticker == 'cash':
            price = 1.0
        else:
            df = self.ticker_dfs[ticker]
            row = df.loc[timestamp]
            price = row.adjusted_close
        return price

    def set_desired(self, timestamp, holdings):
        sell_count = len(self.holdings) + len(holdings) - self.max_holdings
        if sell_count < 0:
            sell_count = 0
        for i in range(sell_count):
            holding = self.holdings[i]
            holding_ticker = holding[0]
            holding_unit = holding[1]
            price = self.get_ticker_price(timestamp, holding_ticker)
            self.cash += holding_unit * price

        del self.holdings[0:sell_count]

        cash_deploy = self.cash / (self.max_holdings - len(self.holdings))
        for ticker in holdings:
            price = self.get_ticker_price(timestamp, ticker)
            holding = (ticker, cash_deploy / price)
            self.holdings.append(holding)
            self.cash -= cash_deploy

    def liquidate(self, timestamp):
        self.set_desired(timestamp, ['cash' for i in range(self.max_holdings)])
        for h in self.holdings:
            self.cash += h[1]
        self.holdings = []


def test_holding(path, pred_filename, max_holding):
    index = pd.read_csv(f'{path}/index.csv')
    ticker_dfs = {}
    for ticker in index.ticker:
        df = get_ticker_df(path, ticker)
        ticker_dfs[ticker] = df
    index = index.append(pd.DataFrame(['cash'], columns=['ticker']), ignore_index=True)

    port = Portfolio(max_holding, ticker_dfs)
    pred = pd.read_csv(f'{path}/{pred_filename}')
    for _, row in pred.iterrows():
        # port.set_desired(row.timestamp, ['xlv'])
        port.set_desired(row.timestamp, [index.iloc[row.target].ticker])

    port.liquidate(pred.iloc[-1].timestamp)
    print(port.cash)

if __name__ == '__main__':
    test_holding('data', 'lgb_pred.csv', 4)
