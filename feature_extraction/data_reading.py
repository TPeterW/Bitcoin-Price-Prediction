#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

def main():
    filepath = sys.argv[1]
    filenames = [filepath + "btc-ohlc-coindesk.csv", filepath + "n-daily-btc-transactions.csv", filepath + "num-btc-transactions-per-block.csv", filepath + "btc-estimated-transaction-volume-blockchain-in-usd.csv"]
    filenames.append(filepath + "fee-cost-per-transaction-percent.csv")
    read_files(filenames)

def read_files(data_files):
    print('hi')

    # Where input data will live:
    data = {}

    for filename in data_files:
        print(filename)
        data_name = filename.split('/')[-1].split('.')[0]
        # set the data col to the filename... cleaner...
        # TODO: Look out for data where there is randomly new data frequencies... sometimes it changes...
        file_data = pd.read_csv(filename, index_col=None, header=0, nrows=1370)
        data[filename] = file_data
        print(file_data)
        # print("data rows:")
        # print(file_data[data_name])
    return data

if __name__ == '__main__':
	main()