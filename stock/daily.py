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

    categories = ['sector', 'size']
    for by in categories:
        index = pd.read_csv(f'data/index_by_{by}.csv')
        predict('data', 'lgb', by, index)
        df = get_last_row(f'data/lgb_{by}_pred.csv')
        print(df.iloc[0].timestamp)
        pick = np.argmax(df.drop('timestamp', axis=1).values[0])
        print(f'\n{index.iloc[pick].values[0]}')

    return


if __name__ == '__main__':
    main()
