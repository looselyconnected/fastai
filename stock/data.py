import argparse
import dateutil
import json
import os
import time
import pdb
import pandas as pd
import urllib

BASEURL="https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&apikey={}&datatype=csv"


def get_ticker_data(ticker, path, key):
    print(f'fetching {ticker}')
    filename = f'{path}/{ticker}.csv'
    url = BASEURL.format(ticker, key)
    try:
        df = pd.read_csv(filename)
    except:
        url += '&outputsize=full'
        df = pd.DataFrame(columns=['timestamp'])

    new_df = pd.read_csv(urllib.request.urlopen(url))

    common_df = pd.merge(new_df, df, how='inner', on='timestamp')
    delta_df = new_df[~new_df.timestamp.isin(common_df.timestamp)]

    delta_df = delta_df.sort_values(by='timestamp')

    delta_csv = delta_df.to_csv(index=False, header=df.empty)
    f = open(filename, 'a')
    f.write(delta_csv)


# we download stock data from quandl if needed and save in the data/downloaded directory,
# then we process them into the processed directory.
def main():
    currentDir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='downloading data')
    parser.add_argument("-o", "--output", help="Set output directory name.", default='data')
    parser.add_argument("-k", "--key", help="Set the alpha advantage api key")

    args = parser.parse_args()
    if args.key == None:
        print('Must have api key')
        return

    path = f'{currentDir}/{args.output}'
    if not os.path.exists(path):
        os.makedirs(path)

    try:
        index = pd.read_csv(f'{path}/index.csv')
    except:
        print('create index.csv first')

    for ticker in index.ticker:
        get_ticker_data(ticker, path, args.key)
        time.sleep(12)


if __name__ == '__main__':
    main()