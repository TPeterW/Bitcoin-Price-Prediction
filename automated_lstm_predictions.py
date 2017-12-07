#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import Embedding
# from keras.layers import LSTM

# TODO: follow the guide....
# helpful reference: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/


# Takes in Dataframes:
def pipeline_lstm(train_frame, labels_frame, cv_train_frame, cv_labels_frame):
    # Importing these within this context, so that they are not loaded every time the pipeline is run:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import LSTM

    # print(train_frame)
    # print(labels_frame)

    train = train_frame.as_matrix()
    labels = labels_frame.as_matrix()

    print(train.shape)
    print(labels.shape)

    # Don't expect there to be timesteps... because features were extracted into timeseries, seperately...
    # Could explore how to build timesteps into this here...

    samples, dimension = train.shape
    train = train.reshape((samples, 1, dimension))

    model = Sequential()
    # input_shape of LSTM first param is time steps...?
    model.add(LSTM(128, input_shape=train.shape[1:]))
    model.add(Dense(10))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(train, labels)


def main():

    # size = 1048576
    size = 104857
    # size = 200000
    split = int(size * .6)

    # to be a param...
    filename = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/data/raw_data/custom_minute_test1.csv"

    file_data = pd.read_csv(filename, index_col=None, header=0, nrows=size)

    matrix = pd.DataFrame.as_matrix(file_data)


    # Setting up model:

    # Need to consider how this number of features could affect overfitting...
    max_features = 2000000
    model = Sequential()
    model.add(Embedding(max_features, output_dim=1))
    input_shape = (810,1)
    # model.add(LSTM(1, input_shape=input_shape, return_sequences=False))

    # model.add(LSTM(128))

    model.add(LSTM(128))

    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])



    x = non_labels[split:size-offset]
    y = y_train[split:]

    # bigger batch sizes makes traning much faster.
    model.fit(x_train_set, y_train_set, batch_size=5, epochs=5, validation_data=(x,y))
    # give the right test set here... labels might be off...
    score = model.evaluate(non_labels[split:size-offset], y_train[split:], batch_size=2)
    print(score)

if __name__ == '__main__':
	main()