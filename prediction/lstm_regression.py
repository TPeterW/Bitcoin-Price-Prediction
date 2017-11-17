#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

# TODO: follow the guide....

def main():
    print('hi')

    # size = 1048576
    size = 10485
    # size = 200000
    split = int(size * .6)

    # Coinbase data set is sparse before row 418053... there are no changes, so it is useles...
    useless_rows = 418053

    filename = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/data/raw_data/custom_minute_test1.csv"

    file_data = pd.read_csv(filename, skiprows=useless_rows, index_col=None, header=0, nrows=size)
    file_data = file_data[file_data.columns[1:]]

    matrix = pd.DataFrame.as_matrix(file_data)

    x_train = matrix[:, 1:12]




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


    # x_train = matrix[:, 1:4]
    # y_train = matrix[:, 4]
    # np.delete(x_train, size, 0)
    # np.delete(x_train, 0, 0)

    # print(x_train)
    # print(y_train)

    # print(file_data)


    # trying offsets...

    x_train_set = non_labels[:split]
    y_train_set = y_train[0:split]
    print(x_train_set)
    print(y_train_set)

    x = non_labels[split:size-offset]
    y = y_train[split:]

    # bigger batch sizes makes traning much faster.
    model.fit(x_train_set, y_train_set, batch_size=5, epochs=5, validation_data=(x,y))
    # give the right test set here... labels might be off...
    score = model.evaluate(non_labels[split:size-offset], y_train[split:], batch_size=2)
    print(score)

if __name__ == '__main__':
	main()