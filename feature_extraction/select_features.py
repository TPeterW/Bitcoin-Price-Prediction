#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import numpy as np
import pandas as pd

from extract_features import convert

from tqdm import tqdm
from tsfresh import extract_features
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from tsfresh.utilities.dataframe_functions import impute

def main():
	if len(sys.argv) < 2:
		print('Usage: ./select_features.py datafile.csv')
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

	sample_windows = pd.DataFrame(columns=timeseries.columns)
	sample_labels = labels.sample(frac=0.01)
	for index in tqdm(sample_labels.index):
		sample_windows = sample_windows.append(timeseries.loc[timeseries['id'] == index])
	sample_windows.reset_index(drop=True, inplace=True)

	sample_features = extract_features(sample_windows, column_id='id', column_sort='time')
	impute(sample_features)

	best_features = rfe(sample_features, sample_labels, 8)
	print(best_features)
	print(type(best_features))
	best_features.savetxt('best_features.csv')

def rfe(features, labels, num_features):
	estimator = Lasso(alpha=200)
	selector = RFE(estimator, num_features, step=10, verbose=10)
	selector = selector.fit(features, labels)

	ranking = np.array(selector.ranking_)
	best_feature_index = np.where(ranking == 1)
	best_features = features.columns[best_feature_index]
	
	return best_features

if __name__ == '__main__':
	main()