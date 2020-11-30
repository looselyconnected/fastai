import argparse
import os
import time
from urllib.request import urlopen
from datetime import date, datetime, timedelta
from io import StringIO

import pandas as pd

BASEURL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&apikey={}&datatype=csv"


class Fields:
    COL_TIME = "timestamp"
    COL_ADJ_CLOSE = "adjusted_close"


# Given a csv file, return a dataframe with just the last row
def get_last_row(filename):
    try:
        statinfo = os.stat(filename)
        f = open(filename)
        header = f.readline()
        f.seek(0 if statinfo.st_size <= 4096 else statinfo.st_size - 4096, 0)
        last_rows = f.read(4096).split("\n")
        last_row = last_rows[-1]
        if len(last_row) == 0:
            last_row = last_rows[-2]
        str = header + last_row
        return pd.read_csv(StringIO(str))
    except:
        return None


# Append the diffs to the end of the file
def append_diff_to_csv(filename, df, fieldname):
    orig = pd.read_csv(filename)
    combined = pd.merge(orig, df, on=df.columns.tolist(), how="outer")
    diff = combined.loc[~combined[fieldname].isin(orig[fieldname])]
    if len(diff) == 0:
        return
    diff_csv = diff.to_csv(header=False, index=False)
    f = open(filename, "a")
    f.write(diff_csv)


def index_to_map(index):
    index_map = {}
    for i in range(len(index)):
        index_map[index.iloc[i].ticker] = i
    index_map["cash"] = len(index_map)
    return index_map


# Returns True if there is new data
def get_ticker_data(ticker, path, key):
    print(f"fetching {ticker}")
    filename = f"{path}/{ticker}.csv"
    url = BASEURL.format(ticker, key)
    try:
        df = pd.read_csv(filename)
    except:
        url += "&outputsize=full"
        df = pd.DataFrame(columns=["timestamp"])

    if len(df) > 0:
        today = date.today()
        last_row_date = datetime.strptime(
            df.timestamp.iloc[-1], "%Y-%m-%d").date()
        if last_row_date == today or (
            last_row_date.isoweekday() == 5
            and today - last_row_date <= timedelta(days=3)
        ):
            return False

    new_df = pd.read_csv(urlopen(url))

    common_df = pd.merge(new_df, df, how="inner", on="timestamp")
    delta_df = new_df[~new_df.timestamp.isin(common_df.timestamp)]

    delta_df = delta_df.sort_values(by="timestamp")

    delta_csv = delta_df.to_csv(index=False, header=df.empty)
    f = open(filename, "a")
    f.write(delta_csv)
    return True


def get_ticker_df(path, ticker, cols=None):
    return pd.read_csv(f"{path}/{ticker}.csv", usecols=cols)


# we download stock data from quandl if needed and save in the data/downloaded directory,
# then we process them into the processed directory.
def main():
    currentDir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description="downloading data")
    parser.add_argument(
        "-o", "--output", help="Set output directory name.", default="data"
    )
    parser.add_argument("-k", "--key", help="Set the alpha advantage api key")

    args = parser.parse_args()
    if args.key == None:
        print("Must have api key")
        return

    path = f"{currentDir}/{args.output}"
    if not os.path.exists(path):
        os.makedirs(path)

    get_all_data(path, args.key)


def get_all_data(path, key):
    # f = urlopen(
    #     "http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/equitypc.csv"
    # )
    # pc_df = pd.read_csv(
    #     f, skiprows=3, names="timestamp,call,put,total,pc_ratio".split(",")
    # )
    # append_diff_to_csv(f"{path}/equitypc.csv", pc_df, "timestamp")

    try:
        index = pd.read_csv(f"{path}/index.csv")
    except:
        print("create index.csv first")

    for ticker in index.ticker:
        got_new_data = get_ticker_data(ticker, path, key)
        if got_new_data:
            # Must slow down to avoid throttling by API server
            time.sleep(12)


if __name__ == "__main__":
    main()
