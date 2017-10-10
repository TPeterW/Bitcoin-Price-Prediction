#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

def main():
	if len(sys.argv) < 2:
		print('Usage: ./extract_features.py datafile.csv')
		exit(1)
	
	filename = sys.argv[1]

	raw_price_data = pd.DataFrame.from_csv(filename, index_col=None, header=0)
	raw_price_data = raw_price_data[raw_price_data.columns[1:]]
	
	timeseries, labels = convert(raw_price_data)
	features = extract_features(timeseries, column_id='id', column_sort='time')
	impute(features)
	features.reset_index(drop=True, inplace=True)
	labels.reset_index(drop=True, inplace=True)
	features.to_csv('features_extracted.csv', sep=',', index=False, header=True)
	labels.to_csv('labels.csv', sep=',', index=False, header=False)

def convert(raw_price_data):
	percent_price_data = pd.DataFrame(index=range(len(raw_price_data)), columns=raw_price_data.columns, dtype=float)

	# convert to fluctuation percentage
	prev = raw_price_data.loc[0]
	for index, row in raw_price_data.iterrows():
		percent_price_data.loc[index]['Open'] = row['Open'] / prev['Open']
		percent_price_data.loc[index]['High'] = row['High'] / prev['High']
		percent_price_data.loc[index]['Low'] = row['Low'] / prev['Low']
		percent_price_data.loc[index]['Close'] = row['Close'] / prev['Close']

		prev = raw_price_data.loc[index]

	labels = percent_price_data['Close'][30:]

	raw = []
	for i in range(30, len(percent_price_data)):
		for j in range(30):
			row = percent_price_data.loc[i - 30 + j].tolist()
			raw.append([i - 30, j, row[0], row[1], row[2], row[3]])

	timeseries = pd.DataFrame(raw, index=None, columns=['id', 'time', 'Open', 'High', 'Low', 'Close'])

	return timeseries, labels

if __name__ == '__main__':
	main()