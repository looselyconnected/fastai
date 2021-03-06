import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from stock.data import get_ticker_df
from stock.train import get_all_delta_data, add_rank_features, add_target


class Portfolio:
    def __init__(self, max_holdings, path, index_name):
        self.holdings = []
        self.cash = 1.0
        self.value = 1.0
        self.max_holdings = max_holdings
        self.histogram = {}
        index = pd.read_csv(f'{path}/{index_name}')
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

        self.value = self.cash
        for holding in self.holdings:
            self.histogram[holding[0]] += 1
            self.value += self.get_ticker_price(timestamp, holding[0]) * holding[1]

    def liquidate(self, timestamp):
        self.set_desired(timestamp, ['cash' for i in range(self.max_holdings)])
        for h in self.holdings:
            self.cash += h[1]
        self.holdings = []
        self.histogram['cash'] -= self.max_holdings

    def get_holding_tickers(self):
        return [h[0] for h in self.holdings]


def test_holding(path, index_name, pred_filenames, max_holding, threshold, confirm_count):
    current = None
    current_count = 0
    res = []
    port = Portfolio(max_holding, path, index_name)
    pred = None
    for filename in pred_filenames:
        if pred is None:
            pred = pd.read_csv(f'{path}/{filename}').set_index('timestamp') / len(pred_filenames)
        else:
            pred += pd.read_csv(f'{path}/{filename}').set_index('timestamp') / len(pred_filenames)

    pred.reset_index(inplace=True)

    for _, row in pred.iterrows():
        row_values = row.drop('timestamp').values
        top_index = np.argmax(row.drop('timestamp').values)

        tickers = None
        if row_values[top_index] >= threshold:
            if current == top_index:
                current_count += 1
            else:
                current_count = 1
                current = top_index

            if current_count >= confirm_count:
                tickers = port.index.iloc[[top_index]].ticker.values.tolist()

        if tickers is None:
            tickers = port.get_holding_tickers()
            if len(tickers) == 0:
                tickers = ['cash']
        port.set_desired(row.timestamp, tickers)
        res += [tickers + [port.value]]

    port.liquidate(pred.iloc[-1].timestamp)
    print(f'\nfrom {pred.iloc[0].timestamp} to {pred.iloc[-1].timestamp} holding {max_holding}'
          f' gain {port.cash}')
    print(port.histogram)
    return pd.DataFrame(res)


def test_holding_target(path, index_name, max_holding, target_days, begin_time, end_time):
    index = pd.read_csv(f'{path}/index.csv')
    df = get_all_delta_data(path, index)
    add_rank_features(df, index)
    add_target(df, target_days, index)
    all_df = df

    port = Portfolio(max_holding, path, index_name)
    begin_index = all_df[all_df.timestamp <= begin_time].index[-1]
    end_index = all_df[all_df.timestamp >= end_time].index[0]

    for _, row in all_df[begin_index : end_index+1].iterrows():
        port.set_desired(row.timestamp, [port.index.iloc[int(row.target)].ticker])

    port.liquidate(end_time)
    print(f'from {begin_time} to {end_time} holding {max_holding}'
          f' target_days {target_days} gain {port.cash}')
    print(port.histogram)


def test_holding_constant(path, index_name, ticker, begin_time, end_time=None):
    port = Portfolio(1, path, index_name)
    if begin_time is None:
        begin_time = port.ticker_dfs[ticker].index[0]

    if end_time is None:
        end_time = port.ticker_dfs[ticker].index[-1]

    port.set_desired(begin_time, [ticker])
    port.liquidate(end_time)
    print(f'from {begin_time} to {end_time} holding {ticker} gain {port.cash}')
    print(port.histogram)

    return port.ticker_dfs[ticker].loc[begin_time:end_time]


def main():
    parser = argparse.ArgumentParser(description='testing performance')
    parser.add_argument("-a", "--algo", help="The algorithm we want to test ")
    parser.add_argument("-b", "--by", help="The breakdown method, segment or size")
    parser.add_argument("-c", "--confirm", help="The number of consecutive selections of a stock to confirm it",
                        default=0)
    parser.add_argument("-s", "--symbol", help="Test just hold the specified symbol")
    parser.add_argument("-k", "--keep", help="How many to keep at one time", default=1)
    parser.add_argument("-t", "--threshold", help="The confidence threshold above which we switch holding",
                        default=0)

    args = parser.parse_args()
    if args.algo is None:
        print('Must have algo name')
        return
    if args.by is None:
        print('Must specify -b segment or size')
        return

    if args.algo == 'all':
        algos = ['lgb', 'nn']
    else:
        algos = [args.algo]

    pred_filenames = []
    for algo in algos:
        pred_filenames += [f'{algo}_{args.by}_pred.csv']

    algo_df = test_holding('data', f'index_by_{args.by}.csv', pred_filenames,
                           int(args.keep), float(args.threshold), int(args.confirm))
    plt.plot(algo_df[1])

    if args.symbol is not None:
        pred_df = pd.read_csv(f'data/{pred_filenames[0]}', nrows=2)
        const_df = test_holding_constant('data', f'index_by_{args.by}.csv', args.symbol, pred_df.iloc[0].timestamp)
        plt.plot(const_df.adjusted_close / const_df.iloc[0].adjusted_close)

    plt.savefig('test_compare.png')
    print('done')


if __name__ == '__main__':
    main()
