#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import pandas as pd

def main():
	if len(sys.argv) < 2:
		print('Usage: ./extract_features.py datafile.csv')
		exit(1)
	
	filename = sys.argv[1]

	raw_price_data = pd.DataFrame.from_csv(filename, index_col=None, header=0)
	raw_price_data = raw_price_data[raw_price_data.columns[1:]]
	
	features = extract(raw_price_data)

	print(features)

def extract(raw_price_data):
	percent_price_data = pd.DataFrame(index=range(len(raw_price_data)), columns=raw_price_data.columns, dtype=float)

	prev = raw_price_data.loc[0]
	for index, row in raw_price_data.iterrows():
		percent_price_data.loc[index]['Open'] = row['Open'] / prev['Open']
		percent_price_data.loc[index]['High'] = row['High'] / prev['High']
		percent_price_data.loc[index]['Low'] = row['Low'] / prev['Low']
		percent_price_data.loc[index]['Close'] = row['Close'] / prev['Close']

		prev = raw_price_data.loc[index]

	return percent_price_data

if __name__ == '__main__':
	main()