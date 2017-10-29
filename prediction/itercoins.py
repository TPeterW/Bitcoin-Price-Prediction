#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

from tsfresh import select_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
	if len(sys.argv) < 3:
		print('Usage: ./predict.py features.csv all_coins.csv')
		exit(1)
	
	all_features = pd.read_csv(sys.argv[1], index_col=None, header=0)
	coins = pd.read_csv(sys.argv[2], index_col=None, header=0)

	print('Finished reading data')

	for coin in coins.columns:
		labels = coins[coin]
		for i in range(len(labels) - 1, 0, -1):
			if labels[i] > labels[i - 1]:
				labels[i] = 1
			else:
				labels[i] = 0

		labels = labels[30:].reset_index(drop=True)

		features = select_features(all_features, labels)
		if len(features.columns) <= 0:
			print('Skipped %s' % coin)
			continue
		accuracy = predict(features, labels, 0.9)
		print(coin, '\t', accuracy)

def predict(features, labels, cutoff):
	cutoff = int(len(features) * cutoff)

	features_train = features.loc[:cutoff]
	labels_train = labels.loc[:cutoff]
	features_test = features.loc[cutoff:].reset_index(drop=True)
	labels_test = labels.loc[cutoff:].reset_index(drop=True)

	lr = LogisticRegression(C=1e5)
	lr.fit(features_train, labels_train)

	predictions = lr.predict(features_test)

	return lr.score(features_test, labels_test)
	# return accuracy_score(labels_test, predictions)

if __name__ == '__main__':
	main()