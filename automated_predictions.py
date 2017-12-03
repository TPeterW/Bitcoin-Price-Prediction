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

from automated_simulation_pipeline import input_file_to_output_name


def main():
    if len(sys.argv) < 3:
        print('Usage: ./predict.py features labels')
        exit(1)

    features = pd.read_csv(sys.argv[1], index_col=None, header=0)
    labels = pd.read_csv(sys.argv[2], index_col=None, header=None, squeeze=True)

    raw_data_name = sys.argv[3]
    minutes_before = int(sys.argv[4])

    predict(features, labels, 0.9, minutes_before, raw_data_name)


# Split this into train and predict?
def predict(features, labels, cutoff, minutes_before, raw_data_name):
    cutoff = int(len(features) * cutoff)

    labels = labels[:len(features)].astype(int)
    features = select_features(features, labels)

    # TODO: move this outside of this file, to the pipeline level:
    features_train = features.loc[:cutoff]
    labels_train = labels.loc[:cutoff]
    features_test = features.loc[cutoff:].reset_index(drop=True)
    labels_test = labels.loc[cutoff:].reset_index(drop=True)



    lr = LogisticRegression(C=1e5)
    lr.fit(features_train, labels_train)

    predictions = lr.predict(features_test)
    score = lr.score(features_test, labels_test)


    # This will be in the pipeline
    raw_data_offset = minutes_before + cutoff



    np.savetxt(input_file_to_output_name(raw_data_name) + '_predictions.csv', predictions, fmt='%i', delimiter=',')

    print('Accuracy: %s' % accuracy_score(labels_test, predictions))
    print('Precision: %s' % precision_score(labels_test, predictions))
    print('Recall: %s' % recall_score(labels_test, predictions))

    # Does not make sense to show this in this part of the code:
    # cnf_matrix = confusion_matrix(labels_test, predictions, [0, 1])
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, normalize=True, classes=[0, 1], title='Nomalized Confusion Matrix')
    # plt.show()


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