#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

def main():
	if len(sys.argv) < 2:
		print('Usage: ./fill_gaps.py file1.csv [file2.csv ...]')
		exit(1)
	
	filenames = sys.argv[1:]

	for filename in filenames:
		filled = fill(filename)
		filled.to_csv(filename[:-4] + '_filled.csv', index=False, header=True)

def fill(filename):
	data = pd.read_csv(filename, index_col=0, header=0)
	data.reset_index(inplace=True, drop=True)

	for col in data.columns:
		for i in range(len(data)):
			if str(data[col][i]) == '*?' and i > 0:
				data[col][i] = data[col][i - 1]

	return data

if __name__ == '__main__':
	main()

