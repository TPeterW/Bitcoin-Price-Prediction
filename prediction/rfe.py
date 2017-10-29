#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

from tsfresh import select_features
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso

def main():
	if len(sys.argv) < 4:
		print('Usage: ./rfe.py features.csv labels.csv num_features')
		exit(1)

	features = pd.read_csv(sys.argv[1], index_col=None, header=0, thousands=',')
	labels = pd.read_csv(sys.argv[2], index_col=None, header=None, squeeze=True)

	features = select_features(features, labels)

	num_features = int(sys.argv[3])

	features.to_csv('features_selected.csv', index=False, header=True)
	return

	best_features = rfe(features, labels, num_features)
	features[best_features].to_csv('features_rfe.csv', index=False, header=True)

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