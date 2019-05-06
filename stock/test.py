import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stock.data import Fields as fld, get_ticker_df, index_to_map
from stock.train import *


class Portfolio:
    def __init__(self, max_holdings, path):
        self.holdings = []
        self.cash = 1.0
        self.max_holdings = max_holdings
        self.histogram = {}
        index = pd.read_csv(f'{path}/index.csv')
        ticker_dfs = {}
        for ticker in index.ticker:
            df = get_ticker_df(path, ticker)
            ticker_dfs[ticker] = df
        self.index = index.append(pd.DataFrame(['cash'], columns=['ticker']), ignore_index=True)
        self.ticker_dfs = ticker_dfs

        for k, v in self.ticker_dfs.items():
            v.set_index('timestamp', inplace=True)
            self.histogram[k] = 0
        self.histogram['cash'] = 0

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

        for holding in self.holdings:
            self.histogram[holding[0]] += 1

    def liquidate(self, timestamp):
        self.set_desired(timestamp, ['cash' for i in range(self.max_holdings)])
        for h in self.holdings:
            self.cash += h[1]
        self.holdings = []
        self.histogram['cash'] -= self.max_holdings


def test_holding(path, pred_filename, max_holding):
    port = Portfolio(max_holding, path)
    pred = pd.read_csv(f'{path}/{pred_filename}')
    for _, row in pred.iterrows():
        # target = np.argmax(row.drop('timestamp').values)
        top_x = 1
        top_index = np.argpartition(row.drop('timestamp').values, -top_x)[-top_x:]
        tickers = port.index.iloc[top_index].ticker.values
        port.set_desired(row.timestamp, tickers)

    port.liquidate(pred.iloc[-1].timestamp)
    print(f'from {pred.iloc[0].timestamp} to {pred.iloc[-1].timestamp} holding {max_holding}'
          f' gain {port.cash}')
    print(port.histogram)


def test_holding_target(path, max_holding, target_days, begin_time, end_time):
    index = pd.read_csv(f'{path}/index.csv')
    df = get_all_delta_data(path, index)
    add_rank_features(df, index)
    add_target(df, target_days, index)
    all_df = df

    port = Portfolio(max_holding, path)
    begin_index = all_df[all_df.timestamp <= begin_time].index[-1]
    end_index = all_df[all_df.timestamp >= end_time].index[0]

    for _, row in all_df[begin_index : end_index+1].iterrows():
        port.set_desired(row.timestamp, [port.index.iloc[int(row.target)].ticker])

    port.liquidate(end_time)
    print(f'from {begin_time} to {end_time} holding {max_holding}'
          f' target_days {target_days} gain {port.cash}')
    print(port.histogram)


def test_holding_constant(path, ticker, begin_time, end_time):
    port = Portfolio(1, path)
    if begin_time is None:
        begin_time = port.ticker_dfs[ticker].iloc[0].timestamp

    if end_time is None:
        end_time = port.ticker_dfs[ticker].iloc[-1].timestamp

    port.set_desired(begin_time, [ticker])
    port.liquidate(end_time)
    print(f'from {begin_time} to {end_time} holding {ticker} gain {port.cash}')
    print(port.histogram)


if __name__ == '__main__':
    test_holding('data', 'lgb_pred.csv', 1)
    # test_holding_target('data', 1, 160, '2013-08-27', '2018-09-05')
    # test_holding_constant('data', 'spy', '2013-08-27', '2019-05-03')