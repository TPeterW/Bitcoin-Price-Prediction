#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import pandas as pd

from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# To Make This A Param
# minutes_before = 30


def main():

    # TODO: move this into a function:
    if len(sys.argv) < 2:
        print('Usage: ./extract_features.py datafile.csv')
        exit(1)

    if not os.path.isfile('timeseries.csv') or not os.path.isfile('labels.csv'):
        filename = sys.argv[1]

        raw_price_data = pd.read_csv(filename, index_col=None, header=0, thousands=',')
        # raw_price_data = raw_price_data[raw_price_data.columns[1:]]		# get rid of the date columns

        timeseries, labels = convert(raw_price_data)

        timeseries.to_csv('timeseries.csv', index=False, header=True)
        labels.to_csv('labels.csv', index=False, header=False)
    else:
        print('Intermediate files exist...')
        timeseries = pd.read_csv('timeseries.csv', index_col=None, header=0)
        labels = pd.read_csv('labels.csv', index_col=None, header=None, squeeze=True)

    features = extract_features(timeseries, column_id='id', column_sort='time')
    impute(features)
    features.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)
    features.to_csv('features_extracted.csv', sep=',', index=False, header=True)
    labels.to_csv('labels.csv', sep=',', index=False, header=False)

    print('Done!')


def convert(raw_price_data, look_back_steps):
    price_data = raw_price_data.astype(float)

    print('Generating labels...')
    close_prices = price_data['Close'].reset_index(drop=True)
    open_prices = price_data['Open'].reset_index(drop=True)

    labels = pd.Series([0] * len(price_data))
    for i in tqdm(range(len(price_data) - 1, look_back_steps - 1, -1)):
        if close_prices[i] > open_prices[i]:
            labels[i] = 1
        else:
            labels[i] = 0
    labels = labels.reset_index(drop=True)[look_back_steps:]

    print('Removing redundent columns...')
    for col in price_data.columns:
        if 'high' in col.lower() or 'low' in col.lower() or 'close' in col.lower():
            price_data.drop(col, axis=1, inplace=True)

    print('Converting into timeseries...')
    raw = []
    for i in tqdm(range(look_back_steps, len(price_data))):
        for j in range(look_back_steps):
            row = price_data.loc[i - look_back_steps + j].tolist()
            raw.append([i - look_back_steps, j] + row)

    timeseries = pd.DataFrame(raw, index=None, columns=['id', 'time'] + price_data.columns.tolist())

    return timeseries, labels

# Needs FileName:
# Minutes_before:


def feature_extraction_process(data_file_name, look_back_steps):

if __name__ == '__main__':
    main()
