#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import configparser
import pandas as pd

import automated_feature_extraction
import automated_predictions

from tsfresh import select_features
from coinutils import input_file_to_output_name

# Auto Extraction Script: In: raw data, minutes_look_back. Out: Extracted Features, Labels (perhaps to change)

# Auto Prediction Script: In: output of extraction and labeling, file names. Out: Predictions, for analysis

# Note:
# This is how you line up the raw_data and the predictions ... +/- 1
# raw_data_offset = minutes_before + cutoff


# maybe more params to specify pipeline options:
def main():
	if len(sys.argv) < 2:
		print('Usage: ./simulation_pipeline.py raw_data')
		exit(1)

	load_params()

	raw_data_name = sys.argv[1]

	file_stem = input_file_to_output_name(raw_data_name)
	print(file_stem)

	return

	# Checking whether feature extraction work has been performed, with the provided input file
	# This will prevent that expensive procedure from occurring, if the pipeline already has the data it would compute
	if not os.path.isfile(file_stem + '_timeseries.csv') or not os.path.isfile(file_stem + '_labels.csv'):
		automated_feature_extraction.feature_extraction_process(raw_data_name, steps_look_back)

	# Reading in the output from the labeling and feature extraction processes:
	features, labels = read_feature_extraction_and_label_output(file_stem + '_features_extracted.csv', file_stem + '_labels.csv')
	features_train, labels_train, features_test, labels_test = split_train_and_test(features, labels)

	# Performing the model training and evaluation:
	automated_predictions.train_and_test_process(features_train, labels_train, features_test, labels_test, file_stem)

# reads the output from the labeling and feature extraction processes:
def read_feature_extraction_and_label_output(features, labels):
	features = pd.read_csv(features, index_col=None, header=0)
	labels = pd.read_csv(labels, index_col=None, header=None, squeeze=True)

	return features, labels


# This takes output from the feature extraction and labeling components:
# Reads csv... returns more convenient data_frames
def split_train_and_test(features, labels):
	global train_test_ratio

	split_point = int(len(features) * train_test_ratio)

	labels = labels[:len(features)].astype(int)
	features = select_features(features, labels)

	features_train = features.loc[:split_point]
	labels_train = labels.loc[:split_point]
	features_test = features.loc[split_point:].reset_index(drop=True)
	labels_test = labels.loc[split_point:].reset_index(drop=True)

	return features_train, labels_train, features_test, labels_test

def load_params():
	config = configparser.ConfigParser()
	config.read('config.ini')

	global LOOKBACK_MINUTES
	LOOKBACK_MINUTES = int(config['Classifiers']['lookback'])

	global TRAIN_TEST_RATIO
	TRAIN_TEST_RATIO = float(config['Training']['ratio'])

if __name__ == '__main__':
	main()
