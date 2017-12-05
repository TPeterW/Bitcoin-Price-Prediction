#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import configparser
import pandas as pd

from tsfresh.feature_extraction import feature_calculators

def main():
	if len(sys.argv) < 2:
		print('Usage: ./extract_best_features.py datafile.csv')
		exit(1)
	
	load_params()

	if not os.path.isfile('timeseries.csv') or not os.path.isfile('labels.csv'):
		filename = sys.argv[1]

		raw_price_data = pd.read_csv(filename, index_col=None, header=0, thousands=',')

		# timeseries, labels = 

def extract_best_features(timeseries, samples_per_window):
	'''
	By RFE
	'''
	extracted_features = pd.DataFrame()
	start = 0
	end = samples_per_window

	kurtosis = []
	count_above_mean = []
	absolute_sum_of_changes = []
	skewness = []
	number_peaks_1 = []
	number_peaks_2 = []
	number_peaks_3 = []
	longest_strike_below_mean = []
	last_location_of_maximum = []

	for i in tqdm(range(len(timeseries) // samples_per_window)):
		window = timeseries[start:end]['contact'].as_matrix().tolist()

		kurtosis.append(feature_calculators.kurtosis(window))
		count_above_mean.append(feature_calculators.count_above_mean(window))
		absolute_sum_of_changes.append(feature_calculators.absolute_sum_of_changes(window))
		skewness.append(feature_calculators.skewness(window))
		number_peaks_1.append(feature_calculators.number_peaks(window, n=1))
		number_peaks_2.append(feature_calculators.number_peaks(window, n=3))
		number_peaks_3.append(feature_calculators.number_peaks(window, n=5))
		longest_strike_below_mean.append(feature_calculators.longest_strike_below_mean(window))
		last_location_of_maximum.append(feature_calculators.last_location_of_maximum(window))

		start = end
		end += samples_per_window

	extracted_features['kurtosis'] = kurtosis
	extracted_features['count_above_mean'] = count_above_mean
	extracted_features['absolute_sum_of_changes'] = absolute_sum_of_changes
	extracted_features['skewness'] = skewness
	extracted_features['number_peaks__n_1'] = number_peaks_1
	extracted_features['number_peaks__n_3'] = number_peaks_2
	extracted_features['number_peaks__n_5'] = number_peaks_3
	extracted_features['longest_strike_below_mean'] = longest_strike_below_mean
	extracted_features['last_location_of_maximum'] = last_location_of_maximum

	return extracted_features

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

def load_params():
	config = configparser.ConfigParser()
	config.read('../config.ini')

	global LOOKBACK_MINUTES
	LOOKBACK_MINUTES = int(config['Classifiers']['lookback'])

if __name__ == '__main__':
	main()