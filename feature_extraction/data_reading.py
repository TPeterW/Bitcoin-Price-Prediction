#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import csv

num_fields = 1
fieldnames = ['date']
num_rows = 1370
output_name = "consolidated_data.csv"


def main():
    if len(sys.argv) >= 2:
        global output_name
        output_name = sys.argv[1]

    # just for testing on my own computer, for now: use ^^^ normally
    filepath = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/raw_data/"
    # filepath will change... provide th appropriate path for your system ... this could synchonize with google docs.

    filenames = [filepath + "btc-ohlc-coindesk.csv", filepath + "n-daily-btc-transactions.csv", filepath + "num-btc-transactions-per-block.csv", filepath + "btc-estimated-transaction-volume-blockchain-in-usd.csv"]
    filenames.append(filepath + "fee-cost-per-transaction-percent.csv")
    # also subject to change^ these are the datasets that I have cleaned.

    write_csv(read_files(filenames))


def read_files(data_files):
    global fieldnames
    global num_fields

    # Structures to hold the data as it is read in:
    dates = pd.read_csv(data_files[0], index_col=None, header=0, nrows=num_rows)
    dates = dates[dates.columns[:1]]
    data_matrix = pd.DataFrame.as_matrix(dates)

    for filename in data_files:
        print(filename)
        # data_name = filename.split('/')[-1].split('.')[0]
        # TODO: Look out for data where there is randomly new data frequencies... sometimes it changes...
        file_data = pd.read_csv(filename, index_col=None, header=0, nrows=num_rows)
        file_data = file_data[file_data.columns[1:]]

        fieldnames.extend(list(file_data.columns))
        num_fields += len(file_data.columns)

        file_matrix = pd.DataFrame.as_matrix(file_data)

        # print(file_matrix)
        data_matrix = np.hstack((data_matrix, file_matrix))

    return data_matrix


def write_csv(data):
    with open(output_name, 'w') as csvfile:
        # w is write only, and it will overwrite an existing file (truncates to 0 length)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(num_rows):
            current_row = {}
            for j in range(num_fields):
                current_row[fieldnames[j]] = data[i][j]
            writer.writerow(current_row)


if __name__ == '__main__':
	main()