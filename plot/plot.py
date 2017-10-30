#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
	if len(sys.argv) < 2:
		print('Usage: ./plot.py coins.csv')
		exit(1)

	coins = pd.read_csv(sys.argv[1], index_col=None, header=0)

	for coin in coins.columns[1:]:
		plt.plot(coins[coin], label=coin[:-6])
	
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()