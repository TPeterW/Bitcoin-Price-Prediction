#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import pandas as pd

def main():
	if len(sys.argv) < 2:
		print('Usage: ./extract_labels.py datafile.csv')
		exit(1)
	filename = sys.argv[1]

	data = pd.read_csv(filename, delimiter=',', index_col=0, header=0, thousands=',')
	data = data.reset_index(drop=True).astype(float)

	for col in data.columns:
		if 'high' in col.lower() or 'low' in col.lower() or 'open' in col.lower():
			data.drop(col, axis=1, inplace=True)

	coins = pd.DataFrame()
	for col in data.columns:
		if 'close' in col.lower():
			coins[col] = data[col]

	coins.to_csv('all_coins.csv', index=False, header=True)

if __name__ == '__main__':
	main()