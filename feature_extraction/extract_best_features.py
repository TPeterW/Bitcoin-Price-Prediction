#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import configparser
import pandas as pd

from tqdm import tqdm
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import feature_calculators

def main():
	if len(sys.argv) < 2:
		print('Usage: ./extract_best_features.py datafile.csv')
		exit(1)
	
	load_params()

	if not os.path.isfile('timeseries.csv') or not os.path.isfile('labels.csv'):
		filename = sys.argv[1]

		raw_price_data = pd.read_csv(filename, index_col=None, header=0, thousands=',')

		timeseries, labels = convert(raw_price_data)

		timeseries.to_csv('timeseries.csv', index=False, header=True)

		labels.reset_index(drop=True, inplace=True)
		labels.to_csv('labels.csv', sep=',', index=False, header=False)
	else:
		print('Intermediate files exist...')
		timeseries = pd.read_csv('timeseries.csv', index_col=None, header=0)
		# timeseries = pd.read_csv('short_timeseries.csv', index_col=None, header=0)
	
	features = extract_best_features(timeseries, samples_per_window=LOOKBACK_MINUTES)
	impute(features)
	features.reset_index(drop=True, inplace=True)
	features.to_csv('features_extracted.csv', sep=',', index=False, header=True)

def extract_best_features(timeseries, samples_per_window):
	'''
	By RFE
	'''
	extracted_features = pd.DataFrame()
	start = 0
	end = samples_per_window

	col_feature1 = []
	col_feature2 = []
	col_feature3 = []
	col_feature4 = []
	col_feature5 = []
	col_feature6 = []
	col_feature7 = []
	col_feature8 = []
	for i in tqdm(range(len(timeseries) // samples_per_window)):
		window = timeseries[start:end]['Open'].as_matrix().tolist()
		col_feature1.append(list(feature_calculators.fft_coefficient(window, [{'coeff': 10, 'attr': 'imag'}]))[0][1])
		col_feature2.append(list(feature_calculators.fft_coefficient(window, [{'coeff': 14, 'attr': 'imag'}]))[0][1])
		col_feature3.append(list(feature_calculators.fft_coefficient(window, [{'coeff': 2, 'attr': 'abs'}]))[0][1])
		col_feature4.append(list(feature_calculators.fft_coefficient(window, [{'coeff': 3, 'attr': 'real'}]))[0][1])
		col_feature5.append(list(feature_calculators.fft_coefficient(window, [{'coeff': 4, 'attr': 'real'}]))[0][1])
		col_feature6.append(list(feature_calculators.fft_coefficient(window, [{'coeff': 6, 'attr': 'imag'}]))[0][1])
		col_feature7.append(list(feature_calculators.fft_coefficient(window, [{'coeff': 7, 'attr': 'imag'}]))[0][1])
		col_feature8.append(list(feature_calculators.fft_coefficient(window, [{'coeff': 8, 'attr': 'real'}]))[0][1])
		
		start = end
		end += samples_per_window

	extracted_features['Open_feature1'] = col_feature1
	extracted_features['Open_feature2'] = col_feature2
	extracted_features['Open_feature3'] = col_feature3
	extracted_features['Open_feature4'] = col_feature4
	extracted_features['Open_feature5'] = col_feature5
	extracted_features['Open_feature6'] = col_feature6
	extracted_features['Open_feature7'] = col_feature7
	extracted_features['Open_feature8'] = col_feature8

	return extracted_features

# 'Volume_(Currency)__fft_coefficient__coeff_10__attr_"imag"'
# 'Volume_(Currency)__fft_coefficient__coeff_14__attr_"imag"'
# 'Volume_(Currency)__fft_coefficient__coeff_2__attr_"abs"'
# 'Volume_(Currency)__fft_coefficient__coeff_3__attr_"real"'
# 'Volume_(Currency)__fft_coefficient__coeff_4__attr_"real"'
# 'Volume_(Currency)__fft_coefficient__coeff_6__attr_"imag"'
# 'Volume_(Currency)__fft_coefficient__coeff_7__attr_"imag"'
# 'Volume_(Currency)__fft_coefficient__coeff_8__attr_"real"'

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