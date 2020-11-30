from datetime import datetime, timedelta
from gc import collect

import argparse
import pandas as pd
import os

ONE_DAY = timedelta(days=1)


def get_all_ticker_data(path, limit=-1):
    index_df = pd.read_csv(f'{path}/index.csv')
    tickers = index_df.ticker

    ticker_dict = {}

    for t in tickers:
        df = pd.read_csv(f'{path}/{t}.csv')
        df.timestamp = pd.to_datetime(df.timestamp)

        adj_close_previous = df.adjusted_close
        adj_close_previous.index += 1
        df = df[['timestamp', 'adjusted_close']].copy()
        df['close_up'] = (df.adjusted_close - adj_close_previous) > 0

        ticker_dict[t] = df.iloc[1:]

        if limit > 0 and len(ticker_dict) >= limit:
            break
    return ticker_dict


def get_all_correlation_data(ticker_dict):
    # For each pair, get the correlation dataframe, which is whether each shared
    # day is a correlation (True or False). Later we can use the sum of the
    # correlation dataframe for a time range to see how correlated the pair is.
    tickers = list(ticker_dict.keys())
    correlation_dict = {}
    for i in range(len(tickers)):
        t1 = tickers[i]

        for j in range(i+1, len(tickers)):
            t2 = tickers[j]

            df1 = ticker_dict[t1]
            df2 = ticker_dict[t2]
            df_join = pd.merge(df1, df2, how='inner', on='timestamp')
            df_join['correlated'] = df_join.close_up_x == df_join.close_up_y

            correlation_dict[f'{t1}-{t2}'] = df_join[['timestamp',
                                                      'correlated']].copy()

    return correlation_dict


def get_correlations(correlation_dict, start_time, end_time):
    l = []
    for key, df in correlation_dict.items():
        if df.timestamp[0] <= start_time:
            # include end_time here, but separate it out
            s = df[(df.timestamp >= start_time) & (df.timestamp <= end_time)]
            next = s.iloc[-1]
            if next.timestamp == end_time:
                l.append((key, s.correlated.iloc[:-1].mean(), next.correlated))

    # Only do it if there are enough correlated stocks
    if len(l) < 200:
        return None, None

    df = pd.DataFrame(l, columns=(['pair', 'ratio', 'next_correlated']))

    df.sort_values('ratio', ascending=False, inplace=True)
    top = df.iloc[:100]

    return top.ratio.mean(), top.next_correlated.mean()


def main():
    currentDir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description="correlation")
    parser.add_argument(
        "-i", "--input", help="Set input directory name.", default="data"
    )

    args = parser.parse_args()

    path = f"{currentDir}/{args.input}"
    if not os.path.exists(path):
        print(f"input path {args.input} not found")
        exit(1)

    ticker_dict = get_all_ticker_data(path, limit=-1)
    correlation_dict = get_all_correlation_data(ticker_dict)
    print('running gc')
    collect()

    # now get the 1000 most correlated pairs based on long term average.
    # this is a performance optimization
    corr_avg_list = []
    for key, df in correlation_dict.items():
        corr_avg_list.append((key, df.correlated.mean()))
    corr_avg_df = pd.DataFrame(corr_avg_list, columns=['pair', 'corr_mean'])
    corr_avg_df.sort_values('corr_mean', ascending=False, inplace=True)
    candidate_pairs = corr_avg_df.iloc[:2000].pair

    corr_candidate_dict = {
        key: correlation_dict[key] for key in candidate_pairs}

    time_s = list(ticker_dict['AMD'].timestamp)
    history = 100

    f = open(f"{currentDir}/correlation.csv", "a")
    f.write('timestamp,past_corr,current_corr\n')
    f.flush()

    for i in range(history, len(time_s)):
        # history window is [start_time, end_time)
        start_time = time_s[i - history]
        end_time = time_s[i]
        print(end_time)

        past_corr, current_corr = get_correlations(
            corr_candidate_dict, start_time, end_time)
        if past_corr:
            f.write(f'{end_time}, {past_corr}, {current_corr}\n')
            f.flush()

        print('running gc')
        collect()

    f.close()
    print('done')


if __name__ == "__main__":
    main()
