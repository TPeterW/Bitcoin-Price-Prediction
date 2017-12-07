#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tsfresh import select_features
import sys

# Takes in Dataframes:
def pipeline_lstm(train_frame, labels_frame, cv_features_frame, cv_labels_frame):
    # Importing these within this context, so that they are not loaded every time the pipeline is run:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import LSTM

    # print(train_frame)
    # print(labels_frame)

    train = train_frame.as_matrix()
    labels = labels_frame.as_matrix()

    cv_features = cv_features_frame.as_matrix()
    cv_labels = cv_labels_frame.as_matrix()

    print(train.shape)
    print(labels.shape)

    # Don't expect there to be timesteps... because features were extracted into timeseries, seperately...
    # Could explore how to build timesteps into this here...

    train = np.expand_dims(train, axis=1)
    cv_features = np.expand_dims(cv_features, axis=1)

    model = Sequential()
    # input_shape of LSTM first param is time steps...?
    model.add(LSTM(128, input_shape=train.shape[1:]))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(train, labels, batch_size=10, epochs=100, validation_data=(cv_features,cv_labels))
    print('LSTM Training Complete')
    score = model.evaluate(cv_features,cv_labels)

    print(score)


# Should import this from automated_simulation_pipeline:
# reads the output from the labeling and feature extraction processes:
def read_feature_extraction_and_label_output(features, labels):
    features = pd.read_csv(features, index_col=None, header=0)
    labels = pd.read_csv(labels, index_col=None, header=None, squeeze=True)

    return features, labels

# This takes output from the feature extraction and labeling components:
# Reads csv... returns more convenient data_frames
def split_train_and_test(features, labels):
    train_test_ratio = 0.6

    split_point = int(len(features) * train_test_ratio)

    labels = labels[:len(features)].astype(int)
    features = select_features(features, labels)

    features_train = features.loc[:split_point]
    labels_train = labels.loc[:split_point]
    features_test = features.loc[split_point:].reset_index(drop=True)
    labels_test = labels.loc[split_point:].reset_index(drop=True)

    return features_train, labels_train, features_test, labels_test

# A commandline runnable version:
def main():
    if len(sys.argv) < 3:
        print('Usage: ./automated_lstm_predictions.py extracted_features labels')
        exit(1)


    # Parse the output of upstream processes:
    features, labels = read_feature_extraction_and_label_output(str(sys.argv[1]), str(sys.argv[2]))
    features_train, labels_train, features_test, labels_test = split_train_and_test(features, labels)
    pipeline_lstm(features_train, labels_train, features_test, labels_test)


if __name__ == '__main__':
	main()