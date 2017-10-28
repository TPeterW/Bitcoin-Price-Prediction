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

	raw_price_data = pd.read_csv(filename, index_col=None, header=0, thousands=',')
	raw_price_data = raw_price_data[raw_price_data.columns[1:]]		# get rid of the date columns

	timeseries, labels = convert(raw_price_data)

	features = extract_features(timeseries, column_id='id', column_sort='time')
	impute(features)
	features.reset_index(drop=True, inplace=True)
	labels.reset_index(drop=True, inplace=True)
	features.to_csv('features_extracted.csv', sep=',', index=False, header=True)
	labels.to_csv('labels.csv', sep=',', index=False, header=False)

	print(features)

def convert(raw_price_data, percentage=False):
	price_data = raw_price_data.astype(float)

	labels = price_data['btc-ohlc-coindesk-Close'][30:]

	for col in price_data.columns:
		if 'high' in col.lower() or 'low' in col.lower() or 'open' in col.lower():
			price_data.drop(col, axis=1, inplace=True)

	raw = []
	for i in range(30, len(price_data)):
		for j in range(30):
			row = price_data.loc[i - 30 + j].tolist()
			raw.append([i - 30, j] + row)

	timeseries = pd.DataFrame(raw, index=None, columns=['id', 'time']+price_data.columns.tolist())

	return timeseries, labels

if __name__ == '__main__':
	main()

# if percentage:
	# 	# convert to fluctuation percentage
	# 	prev = raw_price_data.loc[0]
	# 	for index, row in raw_price_data.iterrows():
	# 		percent_price_data.loc[index]['Open'] = row['Open'] / prev['Open']
	# 		percent_price_data.loc[index]['High'] = row['High'] / prev['High']
	# 		percent_price_data.loc[index]['Low'] = row['Low'] / prev['Low']
	# 		percent_price_data.loc[index]['Close'] = row['Close'] / prev['Close']

	# 		prev = raw_price_data.loc[index]
	# 	price_data = raw_price_data
	# else:
	# 	labels = raw_price_data['Close'][30:]
	# 	price_data = raw_price_data