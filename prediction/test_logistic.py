#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def main():
	if len(sys.argv) < 3:
		print('Usage: ./test_logistic.py features labels')
		exit(1)
	
	features = pd.DataFrame.from_csv(sys.argv[1], index_col=None, header=0)
	labels = pd.Series.from_csv(sys.argv[2], index_col=None, header=None)

	features_train = features.loc[:1300]
	labels_train = labels.loc[:1300]
	features_test = features.loc[1300:]
	labels_test = labels.loc[1300:]

	lr = LinearRegression(copy_X=True, normalize=False)
	lr.fit(features_train, labels_train)

	# predictions = lr.predict(features_test)
	score = lr.score(features_test, labels_test)
	print(score)

if __name__ == '__main__':
	main()