import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# Given a df, add a pct diff feature against a data point of interval ago.
def add_pct_diff_feature(df, column_name, interval):
    df_shift = df.copy()
    df_shift.index += interval
    df[f'{column_name}_pct_diff_{interval}'] = df[column_name] / df_shift[column_name] - 1


# calculate volatility features from the data frames using the column name passed in, adding to the first df.
def get_vol_feature_for_col(src_dfs, colname):
    interval = len(src_dfs)
    sum_s = src_dfs[0][colname].copy()
    for i in range(1, interval):
        sum_s += src_dfs[i][colname]
    mean_s = sum_s / interval
    sum_s = (src_dfs[0][colname] - mean_s) ** 2
    for i in range(1, interval):
        sum_s += (src_dfs[i][colname] - mean_s) ** 2
    std_s = np.sqrt(sum_s / interval)

    return std_s / mean_s

def add_volatility_feature(df, colname, interval):
    if interval > 0:
        dfs = [df]
        for i in range(1, interval):
            df_shift = df[[colname]].copy()
            df_shift.index += i
            dfs.append(df_shift)
    else:
        dfs = [
            df[['high']].rename(columns={'high': colname}),
            df[['low']].rename(columns={'low': colname}),
            df[['open']].rename(columns={'open': colname}),
            df[['close']].rename(columns={'close': colname}),
        ]

    df[f'{colname}_volatility_{interval}'] = get_vol_feature_for_col(dfs, colname)
