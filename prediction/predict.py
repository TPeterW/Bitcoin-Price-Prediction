#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import locale
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from tsfresh import select_features

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# Will make a new script that is organized differently, to accomplish similar goals, within the context of the pipeline.

def main():
	if len(sys.argv) < 3:
		print('Usage: ./predict.py features labels')
		exit(1)
	
	features = pd.read_csv(sys.argv[1], index_col=None, header=0)
	labels = pd.read_csv(sys.argv[2], index_col=None, header=None, squeeze=True)

	# to organize:
	# these should be synchronized with extract_features.py
	raw_data = pd.read_csv(sys.argv[3], index_col=None, header=0, squeeze=True)
	# minutes before:
	minutes_before = int(sys.argv[4])

	predict(features, labels, raw_data, 0.9, minutes_before)

# TODO: also take a raw data parameter, do this in one step with the feature extraction??
# goal with that is to have direct access to the price data... could simply pass as a param...
# This could be handled by one super process, so that it is all known...
# This will occur in the new file. This here is essentially an outline of the new pipeline component. Will remove comments soon.
def predict(features, labels, raw_data, cutoff, minutes_before):
	cutoff = int(len(features) * cutoff)

	labels = labels[:len(features)].astype(int)
	features = select_features(features, labels)

	features_train = features.loc[:cutoff]
	labels_train = labels.loc[:cutoff]
	features_test = features.loc[cutoff:].reset_index(drop=True)
	labels_test = labels.loc[cutoff:].reset_index(drop=True)

	# labels_test = labels.loc[1300:]
	# labels_test = []
	# for i in range(cutoff, len(labels)):
	# 	labels_test.append(1 if labels.loc[i] > labels.loc[i - 1] else 0)

	lr = LogisticRegression(C=1e5)
	lr.fit(features_train, labels_train)

	predictions = lr.predict(features_test)
	score = lr.score(features_test, labels_test)
	
	# actual_prices = labels[cutoff - 1:-1].tolist()
	# for i, val in enumerate(predictions):
	# 	predictions[i] = 1 if val > actual_prices[i] else 0

	# TODO: get the original dataname to write a file, or atleast pass that to the simulation script
	# Perhaps rework how this csv writing occurs, seperate csvs??


	# rows - minutes_before from cutoff onwards shows the correct data...
	# raw_data = raw_data[minutes_before + cutoff:]

	raw_data_offset = minutes_before + cutoff


	num_rows = len(predictions)
	num_fields = 2
	fieldnames = ['predicted', 'true', 'change']
	with open("predictions.csv", 'w') as csvfile:
		# w is write only, and it will overwrite an existing file (truncates to 0 length)
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()

		for i in range(num_rows):
			current_row = {}
			current_row[fieldnames[0]] = predictions[i]
			current_row[fieldnames[1]] = labels_test[i]
			current_row[fieldnames[2]] = raw_data['Change'][i + raw_data_offset]
			writer.writerow(current_row)

	print('Accuracy: %s' % accuracy_score(labels_test, predictions))
	print('Precision: %s' % precision_score(labels_test, predictions))
	print('Recall: %s' % recall_score(labels_test, predictions))

	cnf_matrix = confusion_matrix(labels_test, predictions, [0, 1])
	plt.figure()
	plot_confusion_matrix(cnf_matrix, normalize=True, classes=[0, 1], title='Nomalized Confusion Matrix')
	plt.show()

def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

if __name__ == '__main__':
	main()