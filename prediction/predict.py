#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import locale
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsfresh import select_features

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def main():
	if len(sys.argv) < 3:
		print('Usage: ./predict.py features labels')
		exit(1)
	
	features = pd.read_csv(sys.argv[1], index_col=None, header=0)
	labels = pd.read_csv(sys.argv[2], index_col=None, header=None, squeeze=True)

	# features = select_features(features, labels)

	cutoff = 1100

	features_train = features.loc[:cutoff]
	labels_train = labels.loc[:cutoff]
	features_test = features.loc[cutoff:]

	# labels_test = labels.loc[1300:]
	labels_test = []
	for i in range(cutoff, len(labels)):
		labels_test.append(1 if labels.loc[i] > labels.loc[i - 1] else 0)

	lr = LogisticRegression(C=1e5)
	lr.fit(features_train, labels_train)

	predictions = lr.predict(features_test).tolist()
	# score = lr.score(features_test, labels_test)
	
	actual_prices = labels[cutoff - 1:-1].tolist()
	for i, val in enumerate(predictions):
		predictions[i] = 1 if val > actual_prices[i] else 0

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