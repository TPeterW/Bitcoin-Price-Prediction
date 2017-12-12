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

def main():
    # size = 1048576
    size = 104857
    # size = 200000
    split = int(size * .6)

    offset = 0

    # Coinbase data set is sparse before row 418053... there are no changes, so it is useles...
    useless_rows = 418053

    filename = "/Users/HenrySwaffield/Documents/Middlebury Senior Year/fall/Senior Seminar/project/data/raw_data/coinbaseUSD_1-min_data_test3.csv"

    file_data = pd.read_csv(filename, skiprows=useless_rows, index_col=None, header=0, nrows=size)
    file_data = file_data[file_data.columns[1:]]

    matrix = pd.DataFrame.as_matrix(file_data)

    non_labels = matrix[:, 0:14]
    labels = matrix[:, 14:15]


    print(non_labels.shape)
    print(labels.shape)


    # trying offsets...
    train = np.expand_dims(non_labels, axis=1)

    print(train.shape)

    # train = non_labels

    x_train_set = train[:split]
    y_train_set = labels[0:split]
    print(x_train_set)
    print(y_train_set)

    x = train[split:size-offset]
    y = labels[split:]


    model = Sequential()
    # input_shape of LSTM first param is time steps...?
    model.add(LSTM(64, input_shape=train.shape[1:]))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])




    # bigger batch sizes makes traning much faster.
    model.fit(x_train_set, y_train_set, batch_size=20, epochs=10, validation_data=(x,y))
    # give the right test set here... labels might be off...
    print('[loss, accuracy]:')
    score = model.evaluate(x, y, batch_size=200)
    print(score)

if __name__ == '__main__':
	main()
