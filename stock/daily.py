import subprocess
import os
import argparse
import pandas as pd
import numpy as np

from common.data import get_last_row
from stock.data import get_all_data
from stock.pred import predict

def main():
    parser = argparse.ArgumentParser(description='daily run and reporting')
    parser.add_argument("-k", "--key", help="Set the alpha advantage api key")

    args = parser.parse_args()
    if args.key == None:
        print('Must have api key')
        return
    get_all_data('data', args.key)

    threshold = {'sector': 0.5, 'size': 0.6}
    for by in threshold:
        index = pd.read_csv(f'data/index_by_{by}.csv')
        predict('data', 'lgb', by, index)
        df = get_last_row(f'data/lgb_{by}_pred.csv')
        print(df.iloc[0].timestamp)
        df = df.drop('timestamp', axis=1)
        pick = np.argmax(df.values[0])
        print(f'{index.iloc[pick].values[0]} probability {df.iloc[0].values[pick]} thresold {threshold[by]}\n\n')

    return


if __name__ == '__main__':
    main()
