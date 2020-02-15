import numpy as np
import pandas as pd

# Given a df, add a pct diff feature against a data point of interval ago.
def add_pct_diff_feature(df, column_name, interval):
    df_shift = df.copy()
    df_shift.index += interval
    df[f'{column_name}_pct_diff_{interval}'] = df[column_name] / df_shift[column_name] - 1


# Add the volatility feature
def add_volatility_feature(df, column_name, interval):
    for i in range(interval - 1, len(df)):
        df.loc[i, f'{column_name}_volatility_{interval}'] = df.loc[i - interval + 1: i, column_name].std() / \
                                                            df.loc[i - interval + 1: i, column_name].mean()
