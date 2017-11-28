#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import pandas as pd

from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

minutes_before = 30

def main():
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

def convert(raw_price_data, percentage=False):
	price_data = raw_price_data.astype(float)

	print('Generating labels...')
	close_prices = price_data['Close'].reset_index(drop=True)
	open_prices = price_data['Open'].reset_index(drop=True)

	labels = pd.Series([0] * len(price_data))
	for i in tqdm(range(len(price_data) - 1, minutes_before - 1, -1)):
		if close_prices[i] > open_prices[i]:
			labels[i] = 1
		else:
			labels[i] = 0
	labels = labels.reset_index(drop=True)[minutes_before:]

	print('Removing redundent columns...')
	for col in price_data.columns:
		if 'high' in col.lower() or 'low' in col.lower() or 'close' in col.lower():
			price_data.drop(col, axis=1, inplace=True)

	print('Converting into timeseries...')
	raw = []
	for i in tqdm(range(minutes_before, len(price_data))):
		for j in range(minutes_before):
			row = price_data.loc[i - minutes_before + j].tolist()
			raw.append([i - minutes_before, j] + row)

	timeseries = pd.DataFrame(raw, index=None, columns=['id', 'time'] + price_data.columns.tolist())

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